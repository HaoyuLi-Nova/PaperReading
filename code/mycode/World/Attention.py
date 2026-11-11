import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(x, positions, base=10000.0):
    """
    x: (B, H, L, D)，D 必须为偶数
    positions: (L,) 或 (B, L)，整数位置
    返回旋转后的 x，shape 不变
    """
    B, H, L, D = x.shape
    assert D % 2 == 0, "RoPE 要求 head_dim 为偶数"
    if positions.dim() == 1:
        positions = positions.unsqueeze(0).expand(B, -1)   # (B, L)

    # 频率：θ_i = base^(-2i/D), i = 0..D/2-1
    i = torch.arange(0, D, 2, device=x.device, dtype=torch.float32)  # (D/2,)
    theta = base ** (-i / D)                                          # (D/2,)
    # 角度：m·θ_i，shape (B, L, D/2)
    angles = positions.float().unsqueeze(-1) * theta.unsqueeze(0).unsqueeze(0)
    cos = angles.cos().unsqueeze(1)   # (B, 1, L, D/2)
    sin = angles.sin().unsqueeze(1)

    # 两两分组
    x1 = x[..., 0::2]   # (B, H, L, D/2)
    x2 = x[..., 1::2]
    # 旋转
    out = torch.stack([x1 * cos - x2 * sin,
                       x1 * sin + x2 * cos], dim=-1)  # (B, H, L, D/2, 2)
    return out.flatten(-2)  # (B, H, L, D)


class MultiHeadAttention(nn.Module):
    """
    支持：
      - self-attn / cross-attn（通过传入不同的 kv）
      - causal mask
      - padding mask（key 侧）
      - RoPE（仅 self-attn 的 Q、K）
    """
    def __init__(self, d_model, n_head, dropout=0.0, use_rope=False, rope_base=10000.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.use_rope = use_rope
        self.rope_base = rope_base

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.n_head, self.head_dim).transpose(1, 2)  # (B,H,L,D)

    def _merge_heads(self, x):
        B, H, L, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * D)          # (B,L,d_model)

    def forward(self, q, k=None, v=None,
                attn_mask=None,        # (Lq, Lk) causal，或 (B,1,Lq,Lk) 任意 additive
                key_padding_mask=None, # (B, Lk) bool，True=有效
                q_positions=None, k_positions=None):
        """
        q: (B, Lq, d_model)
        k, v: (B, Lk, d_model)；若为 None 则做 self-attn（k=v=q）
        """
        if k is None:
            k = q
        if v is None:
            v = k

        B, Lq, _ = q.shape
        Lk = k.shape[1]

        # 1) 投影 + 拆头
        Q = self._split_heads(self.w_q(q))   # (B,H,Lq,D)
        K = self._split_heads(self.w_k(k))   # (B,H,Lk,D)
        V = self._split_heads(self.w_v(v))   # (B,H,Lk,D)

        # 2) RoPE（只对 self-attn 的 Q、K）
        if self.use_rope and k is q:         # 用 is 判断是否 self-attn
            if q_positions is None:
                q_positions = torch.arange(Lq, device=q.device)
            if k_positions is None:
                k_positions = q_positions
            Q = apply_rope(Q, q_positions, self.rope_base)
            K = apply_rope(K, k_positions, self.rope_base)

        # 3) 打分
        scale = 1.0 / math.sqrt(self.head_dim)   # 注意：head_dim，不是 d_model
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale   # (B,H,Lq,Lk)

        # 4) 组装 mask（additive，-inf 表示屏蔽）
        additive = torch.zeros(1, device=q.device)
        if attn_mask is not None:
            # 转成 additive：True/1 表示屏蔽，False/0 表示保留
            am = attn_mask
            if am.dtype == torch.bool:
                am = torch.zeros_like(am, dtype=scores.dtype).masked_fill_(am, float('-inf'))
            additive = additive + am
        if key_padding_mask is not None:
            # (B, Lk) bool，True=有效 → 屏蔽 False
            kpm = (~key_padding_mask).to(scores.dtype) * float('-inf')  # (B,Lk)
            additive = additive + kpm.view(B, 1, 1, Lk)
        if not torch.is_tensor(additive) or additive.numel() > 1:
            scores = scores + additive

        # 5) softmax + dropout
        attn = F.softmax(scores, dim=-1)
        # 处理全 -inf 行（比如 padding query），避免 NaN
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # 6) 加权求和 + 合并头 + 输出投影
        out = torch.matmul(attn, V)          # (B,H,Lq,D)
        out = self._merge_heads(out)         # (B,Lq,d_model)
        return self.w_o(out)


# ---------- 用法示例 ----------
if __name__ == "__main__":
    B, L, d_model, H = 2, 6, 64, 4
    x = torch.randn(B, L, d_model)

    # self-attn + causal mask + RoPE
    causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)  # 上三角=True=屏蔽
    mha = MultiHeadAttention(d_model, H, use_rope=True)
    y = mha(x, attn_mask=causal)
    print("self-attn + causal + rope:", y.shape)   # (2, 6, 64)

    # cross-attn：Q 来自 y (长度 4)，K/V 来自 x (长度 6)
    yq = torch.randn(B, 4, d_model)
    kpm = torch.ones(B, L, dtype=torch.bool)       # 全部有效
    y2 = mha(yq, k=x, v=x, key_padding_mask=kpm)
    print("cross-attn:", y2.shape)                 # (2, 4, 64)