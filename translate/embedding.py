from torch import nn
import torch
import math
from dataset import de_vocab, de_process, train_dataset


class EmbeddingWithPosition(nn.Module):

    def __init__(self, vocab_size, emb_size, dropout=0.1, seq_max_len=5000):
        super().__init__()

        self.seq_emb = nn.Embedding(vocab_size, emb_size)

        position_idx = torch.arange(0, seq_max_len, dtype=torch.float).unsqueeze(-1)
        position_emb_fill = position_idx * torch.exp(-torch.arange(0, emb_size,2) * math.log(10000.0) / emb_size)
        pos_encoding = torch.zeros(seq_max_len, emb_size)
        pos_encoding[:, 0::2] = torch.sin(position_emb_fill)
        pos_encoding[:, 1::2] = torch.cos(position_emb_fill)
        self.register_buffer('pos_encoding', pos_encoding)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.seq_emb(x)
        x += self.pos_encoding.unsqueeze(0)[:, :x.size()[1],:]
        return self.dropout(x)


if __name__ == "__main__":
    emb = EmbeddingWithPosition(len(de_vocab), 128)

    de_tokens, de_ids = de_process(train_dataset[0][0])

    print(de_tokens, de_ids)

    de_ids_tensor = torch.tensor(de_ids, dtype=torch.long)

    emb_result = emb(de_ids_tensor.unsqueeze(0))
    print("de_ids_tensor", de_ids_tensor.size(), 'emb_result', emb_result.size())