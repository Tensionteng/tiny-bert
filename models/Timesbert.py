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

        # 整个数据集的协方差矩阵
        self.cov_matrix = getattr(configs, 'cov_matrix', None)

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


