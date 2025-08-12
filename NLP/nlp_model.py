import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyPromptEncoder(nn.Module):
    def __init__(self, depth_vocab_size, purpose_vocab_size, embedding_dim=16):
        super().__init__()
        self.depth_embed = nn.Embedding(depth_vocab_size, embedding_dim)
        self.purpose_embed = nn.Embedding(purpose_vocab_size, embedding_dim)

        # 融合后投射到频段权重维度，比如4个频段权重
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # 4个小波频段权重，或可扩展
            nn.Sigmoid()       # 归一化到 [0,1]
        )

    def forward(self, depth_ids, purpose_ids):
        # depth_ids, purpose_ids: [batch_size],类别索引
        depth_emb = self.depth_embed(depth_ids)    # [B, embedding_dim]
        purpose_emb = self.purpose_embed(purpose_ids)  # [B, embedding_dim]

        combined = torch.cat([depth_emb, purpose_emb], dim=-1)  # [B, 2*embedding_dim]

        weights = self.fc(combined)  # [B, 4], 每个频段权重

        return weights  # 返回频段权重

# 使用示例
depth_vocab = {"shallow":0, "middle":1, "deep":2}
purpose_vocab = {"structure":0, "interface":1, "anomaly":2}

encoder = TinyPromptEncoder(depth_vocab_size=3, purpose_vocab_size=3, embedding_dim=16)

depth_ids = torch.tensor([depth_vocab["shallow"], depth_vocab["deep"]])  # batch=2
purpose_ids = torch.tensor([purpose_vocab["structure"], purpose_vocab["anomaly"]])

freq_weights = encoder(depth_ids, purpose_ids)  # [2,4]
print(freq_weights)
