import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention

# Gated Self Attention（head-specific 版本）
class GatedRobertaSelfAttention(RobertaSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        # gate 投影层，从 hidden_size -> num_heads（head-specific scalar）
        self.gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads)
        # 初始化：使初始 gate ≈ 0.88 ，保留预训练能力（和qwen的源代码的有区别）
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        if self.gate_proj.bias is not None:
            nn.init.constant_(self.gate_proj.bias, 2.0)
            # nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        **kwargs,
    ):

        # 兼容性处理
        kwargs.pop("cache_position", None)
        kwargs.pop("position_bias", None)


        # 原始 attention 计算
        self_outputs = super().forward(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            output_attentions,
            **kwargs,
        )
        attention_output = self_outputs[0]  # (B, S, hidden)

        # 计算 head-specific gate
        # 从 hidden_states (query-like) 投影得到 (B, S, num_heads)
        gate_logits = self.gate_proj(hidden_states)  # (B, S, num_heads)
        gate = torch.sigmoid(gate_logits)  # (B, S, num_heads)

        # 这里的 S (Sequence Length) 在 iTransformer 里代表 "变量数量"
        # 我们希望针对每个变量、每个头单独控制
        # Reshape attention_output: (B, S, Heads, Head_Dim)
        B, S, H = gate.shape
        head_dim = attention_output.shape[-1] // H
        
        attn_per_head = attention_output.view(B, S, H, head_dim)

        # Reshape gate for broadcasting: (B, S, Heads, 1)
        gate_broadcast = gate.unsqueeze(-1)


        gated_attn = attn_per_head * gate_broadcast
        
        # 恢复形状
        gated_output = gated_attn.reshape(B, S, -1)

        return (gated_output,) + self_outputs[1:]

# 自定义带 Gated Attention 的模型
class GatedRobertaModel(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        # 替换所有层的 self-attention 为 gated 版本
        for layer in self.encoder.layer:
            layer.attention.self = GatedRobertaSelfAttention(config)

        # 初始化新参数
        self.post_init()  # 初始化

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # 加载 config
        config = AutoConfig.from_pretrained("llm/roberta-tiny")
        # 创建自定义 gated 模型（不加载原权重）
        self.bert = GatedRobertaModel(config)
        # 加载原预训练权重（gate_proj 是新参数，会随机初始化）
        original_bert = AutoModel.from_pretrained("llm/roberta-tiny")
        missing, unexpected = self.bert.load_state_dict(original_bert.state_dict(), strict=False)  # 忽略新参数
        print(f"Loaded weights. Missing (new gates): {len(missing)}")

        if hasattr(config, "hidden_size"):
            d_model = config.hidden_size
        else:
            d_model = config.d_model  # 视模型而定

        in_d_model = d_model  # 你的代码逻辑
        self.fc = nn.Linear(self.seq_len, in_d_model)
        self.proj = nn.Linear(d_model, self.pred_len)

        if hasattr(config, "max_position_embeddings"):
            self.max_chunk_size = config.max_position_embeddings - 2
        else:
            self.max_chunk_size = 1024

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        # ========== Normalization (from iTransformer) ==========
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        
        _, _, N = x.shape  # N = num_features
        
        x = x.permute(0, 2, 1)  # (B, T, F) -> (B, F, T)
        x = self.fc(x)  # (B, F, hidden)

        chunked_outputs = []
        for i in range(0, x.size(1), self.max_chunk_size):
            chunk = x[:, i : i + self.max_chunk_size, :]
            bert_output = self.bert(inputs_embeds=chunk).last_hidden_state  # (B, chunk_len, hidden)
            chunked_outputs.append(bert_output)

        bert_output = torch.cat(chunked_outputs, dim=1)  # (B, F, hidden)
        proj_output = self.proj(bert_output)  # (B, F, pred_len)
        output = proj_output.transpose(1, 2)  # (B, pred_len, F)
        
        # ========== De-Normalization ==========
        output = output * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        output = output + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        
        return output