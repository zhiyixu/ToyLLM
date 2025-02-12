from torch import nn 
import torch
import math
from dataset import de_vocab, de_process, train_dataset
from embedding import EmbeddingWithPosition 

class MultiHeadAttention(nn.Module):

    def __init__(self, emb_size, q_k_size, v_size, head):
        super().__init__() 

        self.emb_size = emb_size
        self.q_k_size = q_k_size 
        self.v_size = v_size
        self.head = head 

        self.w_q = nn.Linear(emb_size, head * q_k_size)
        self.w_k = nn.Linear(emb_size, head * q_k_size)
        self.w_v = nn.Linear(emb_size, head * v_size)

    def forward(self, x_q, x_k_v, attn_mask):

        q = self.w_q(x_q)
        k = self.w_k(x_k_v)

        q = q.view(q.size()[0], q.size()[1], self.head, self.q_k_size).transpose(1,2) 
        k = k.view(k.size()[0], k.size()[1], self.head, self.q_k_size).transpose(1,2).transpose(2,3) 

        attn = torch.matmul(q, k) / math.sqrt(self.q_k_size)  # 注意力分值 

        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.head, -1, -1)
        attn = attn.masked_fill(attn_mask, -1e9)
        attn = torch.softmax(attn, dim=-1)

        v = self.w_v(x_k_v)
        v = v.view(v.size()[0], v.size()[1], self.head, self.v_size).transpose(1,2) 
        z = torch.matmul(attn, v) 
        z = z.transpose(1,2)

        return z.reshape(z.size()[0], z.size()[1], -1)
    
if __name__ == "__main__":
    emb = EmbeddingWithPosition(len(de_vocab), 128)
    de_tokens, de_ids = de_process(train_dataset[1][0])
    de_ids_tensor = torch.tensor(de_ids, dtype=torch.long)
    emb_result = emb(de_ids_tensor.unsqueeze(0))
    print("de_ids_tensor", de_ids_tensor.size(), 'emb_result', emb_result.size())

    multihead= MultiHeadAttention(emb_size=128, q_k_size=256, v_size=128, head=8)
    attn_mask = torch.zeros((1, de_ids_tensor.size()[0], de_ids_tensor.size()[0]), dtype=torch.bool)
    multihead_result = multihead.forward(emb_result, emb_result, attn_mask)
    print("multihead_result", multihead_result.size())