import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import transformers as tfs
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # self.bert = tfs.AutoModel.from_pretrained("prajjwal1/bert-tiny")
        # self.bert = tfs.AutoModel.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2")
        # self.bert = tfs.AutoModel.from_pretrained("alibaba-pai/pai-bert-tiny-zh")
        # self.bert = tfs.AutoModel.from_pretrained("prajjwal1/bert-small") # traffic 0.393 1e-4
        # self.bert = tfs.AutoModel.from_pretrained("microsoft/deberta-v3-small")
        self.bert = tfs.AutoModel.from_pretrained(
            "llm/roberta-tiny"
        )  # traffic best 32 1e-4
        # self.bert = tfs.AutoModel.from_pretrained("anhleHGF/tiny-electra-sst2-distilled")
        # self.bert = tfs.AutoModel.from_pretrained("TinyPixel/gpt2-40m")

        # d_model = in_d_model = self.bert.config.n_embd
        if hasattr(self.bert.config, "hidden_size"):
            d_model = self.bert.config.hidden_size
        else:
            d_model = self.bert.config.d_model

        if hasattr(self.bert.config, "embedding_size"):
            in_d_model = self.bert.config.embedding_size
        else:
            in_d_model = d_model

        self.fc = nn.Linear(self.seq_len, in_d_model)
        self.proj = nn.Linear(d_model, self.pred_len)
        if hasattr(self.bert.config, "max_position_embeddings"):
            self.max_chunk_size = self.bert.config.max_position_embeddings - 2
        else:
            self.max_chunk_size = 1024

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        # Input: (B, T, F)
        x = x.permute(0, 2, 1)
        x = self.fc(x)

        # Process in chunks if sequence length is too large
        chunked_outputs = []
        for i in range(0, x.size(1), self.max_chunk_size):
            chunk = x[:, i : i + self.max_chunk_size, :]  # (B, chunk_size, H)
            bert_output = self.bert(inputs_embeds=chunk)[0]  # (B, chunk_size, H)
            chunked_outputs.append(bert_output)

        # Concatenate all chunk outputs
        bert_output = torch.cat(chunked_outputs, dim=1)  # (B, F(token), H)

        # Project to (B, F, pred_len)
        proj_output = self.proj(bert_output)

        # Transpose back to (B, pred_len, F)
        output = proj_output.transpose(1, 2)

        return output


# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()

#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in

#         # --- 1. 同态映射 (Φ) & PCA先验注入 ---
#         raw_pca = getattr(configs, "pca_components", None)
#         if isinstance(raw_pca, (torch.Tensor, np.ndarray)):
#             self.pca_components = torch.from_numpy(raw_pca).float() if isinstance(raw_pca, np.ndarray) else raw_pca.float()
#             self.m_components = self.pca_components.shape[0]
#         else:
#             self.pca_components = None
#             self.m_components = self.enc_in # 如果没有PCA，则不降维

#         self.phi_mapping = nn.Linear(self.enc_in, self.m_components, bias=False)
#         if self.pca_components is not None:
#             print(f"Initializing phi_mapping with PCA components (m={self.m_components}).")
#             self.phi_mapping.weight.data.copy_(self.pca_components)
#         else:
#             print("No PCA components provided, phi_mapping is randomly initialized.")

#         # --- 2. Tiny-BERT 编码器 ---
#         # (选择一个小的双向模型)
#         self.bert = tfs.AutoModel.from_pretrained("roberta-tiny")
#         # self.bert = tfs.AutoModel.from_pretrained("prajjwal1/bert-tiny") # bert-tiny 更符合论文思想
#         d_model = self.bert.config.hidden_size

#         # --- 3. 投影与解码器层 (新) ---
#         # 新增: 将 m 维的潜在因子投影到 BERT 的 d_model 维
#         self.input_projection = nn.Linear(self.m_components, d_model)
#         self.input_projection = nn.Linear(self.seq_len, d_model)

#         # 修改: 定义一个线性解码器，将BERT的输出表示解码为预测结果
#         self.decoder = nn.Linear(d_model, self.pred_len * self.enc_in)

#     def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
#         # 输入 x: (Batch, seq_len, enc_in) -> (B, T, d)

#         # 1. 应用 Φ 映射: (B, T, d) -> (B, T, m)
#         z = self.phi_mapping(x)

#         # 2. 投影到BERT的输入维度: (B, T, m) -> (B, T, d_model)
#         z = z.permute(0, 2, 1).contiguous()  # (B, m, T)
#         z_projected = self.input_projection(z)

#         # 3. 通过Tiny-BERT进行时间编码 (注意力作用在时间T上)
#         # last_hidden_state.shape: (B, T, d_model)
#         bert_output = self.bert(inputs_embeds=z_projected).last_hidden_state

#         # 4. 提取序列的最终表示 (如论文所述，使用最后一个时间步h_T)
#         # representation.shape: (B, d_model)
#         representation = bert_output[:, -1, :]

#         # 5. 解码生成预测
#         # forecast.shape: (B, pred_len * enc_in)
#         forecast = self.decoder(representation)

#         # 6. Reshape回标准的时间序列格式
#         # (B, pred_len * enc_in) -> (B, pred_len, enc_in)

#         return forecast.view(forecast.size(0), self.pred_len, self.enc_in).transpose(1, 2)
