import math
import torch
import torch.nn as nn

class MultiHeadSelfAttentionFormal(nn.Module):
    def __init__(self, hidden_dim=2048, head_num=8, attn_dropout = 0.1, out_dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num

        self.attn_drop = nn.Dropout(attn_dropout)
        self.out_drop = nn.Dropout(out_dropout)

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, X, mask=None):
        #X [B, N, D]
        batch_size, seq_len, _ = X.size()
        Q = self.W_q(X).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1,2) # (B,H,N,D) 
        K = self.W_k(X).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1,2) 
        V = self.W_v(X).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1,2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim) # (B,H,N,D) @ (B,H,D,N) = (B,h,N,N)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1) # (B,H,N,N)
        attn = self.attn_drop(attn)

        # 3) 加权求和 + 合并头 + 
        out = attn @ V # (B,H,N,N) @ (B,H,N,Dk) = (B,H,N,Dk)
        out = out.transpose(1,2).reshape(batch_size,seq_len,self.hidden_dim) # (B,N,H,Dk)
        out = self.W_o(out)
        out = self.out_drop(out)
        return out

def main():
    device = torch.device("cuda:2")
    model = MultiHeadSelfAttentionFormal().to(device)
    input = torch.rand([4,1024,2048], device=device)
    output = model.forward(input)

    torch.save(model.state_dict,"model.pt")
    print("model saved")

    checkpoint = torch.load("model.pt", map_location = "cuda")
    model.load_state_dict(checkpoint)

    print(output.shape)
    return 0

if __name__ == "__main__":
    main()