import torch
import torch.nn as nn
import torch.cuda.amp as amp
import math

from .attention import flash_attention
from common.constants import JOINT_NUM

__all__ = ['Transformer']

#_______________________________________________modules___________________________________________

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

# rope params with stride
@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000, stride=1, start_idx=0):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(start_idx, max_seq_len+start_idx) * stride ,
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

@amp.autocast(enabled=False)
def rope_apply_motion(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), 2 * (c // 3)], dim=1)

    # loop over samples
    output = []
    for i, (f, l, _) in enumerate(grid_sizes.tolist()):
        seq_len = f * l

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, -1).expand(f, l, -1),
            freqs[1][:l].view(1, l, -1).expand(f, l, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

@amp.autocast(enabled=False)
def rope_apply_1d(x, grid_sizes, freqs):
    # 3D grid size with 1D frequencies
    n, c = x.size(2), x.size(3) // 2

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()): # h=w=1, f=T
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = freqs[:f].view(f, 1, 1, -1).expand(f, h, w, -1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class LayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)
    
    def forward(self, x):
        return super().forward(x.float()).type_as(x)
    
#________________________________________Video Attnention Blocks__________________________________________

class SelfAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v
        
        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CrossAttention(SelfAttention):

    def forward(self, x, context, context_lens):
        """
        x:              [B, L1, C].
        context:        [B, L2, C].
        context_lens:   [B].
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class AttentionBlock(nn.Module):
    
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = LayerNorm(dim, eps)
        self.self_attn = SelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = LayerNorm(
            dim, eps, elementwise_affine=True
        ) if cross_attn_norm else nn.Identity()
        self.cross_attn = CrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = LayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
    
    def forward(
        self,
        x,
        e,
        e_m,
        seq_lens_video,
        seq_lens_motion,
        grid_sizes_motion,
        grid_sizes_video,
        freqs,
        freqs_motion,
        context,
        context_motion,
        context_lens,
        seq_len_motion, 
        seq_len_video,
        forward_type='video_with_motion',
        task_embedding_motion=None,
        task_embedding_video=None,
    ):
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        bsz = x.shape[0]

        def detach_seq(x):
            video_list, motion_list = [], []
            for i in range(bsz):
                sl_v = seq_lens_video[i].item()
                sl_m = seq_lens_motion[i].item()

                # Ensure non-negative padding lengths for torch.split
                v_pad_len = max(0, seq_len_video - sl_v)
                m_pad_len = max(0, seq_len_motion - sl_m)

                # Assuming x[i:i+1] total length matches sum of split_sizes
                # i.e., x is structured as [v_valid, m_valid, v_pad, m_pad]
                split_sizes = [sl_v, sl_m, v_pad_len, m_pad_len]
                
                v_valid, m_valid, v_pad, m_pad = torch.split(
                    x[i:i+1], 
                    split_sizes, # Use calculated split_sizes
                    dim=1
                )

                v_cur = torch.cat([v_valid, v_pad], dim=1)
                m_cur = torch.cat([m_valid, m_pad], dim=1)
                video_list.append(v_cur)
                motion_list.append(m_cur)
            x_video, x_motion = torch.cat(video_list, dim=0), torch.cat(motion_list, dim=0)
            return x_video, x_motion
        
        def concat_seq(x_video, x_motion):
            x_list = []
            for i in range(bsz):
                sl_v = seq_lens_video[i].item()
                sl_m = seq_lens_motion[i].item()

                # Ensure non-negative padding lengths for torch.split
                v_pad_len = max(0, seq_len_video - sl_v)
                m_pad_len = max(0, seq_len_motion - sl_m)

                # x_video and x_motion here are assumed to be fixed-length (seq_len_video, seq_len_motion)
                # and structured as [valid_part, padding_part]
                x_v, x_v_pad = torch.split(x_video[i:i+1], [sl_v, v_pad_len], dim=1)
                x_m, x_m_pad = torch.split(x_motion[i:i+1], [sl_m, m_pad_len], dim=1)
                
                # Re-concatenate into the original [v_valid, m_valid, v_pad, m_pad] structure
                x_list.append(torch.cat([x_v, x_m, x_v_pad, x_m_pad], dim=1))
            return torch.cat(x_list, dim=0)
        
        # detach the input sequence
        x, x_m = detach_seq(x)

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens_video, grid_sizes_video, freqs
        )
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        
        # concat the video and motion sequence
        x = concat_seq(x, x_m)
        
        return x


# _______________________________________ Multi-Model Tranformer Blocks ________________________________________

class JointSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # q k v
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        
        self.motion_q = nn.Linear(dim, dim)
        self.motion_k = nn.Linear(dim, dim)
        self.motion_v = nn.Linear(dim, dim)
        self.motion_o = nn.Linear(dim, dim)

        # nrom
        norm_layer = lambda: RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_q = norm_layer()
        self.norm_k = norm_layer()
        self.motion_norm_q = norm_layer()
        self.motion_norm_k = norm_layer()

    def _split_video_motion(self, x, seq_lens_video, seq_lens_motion, seq_len_video, seq_len_motion):
        """detach video and motion features from the input tensor x"""
        batch_size = x.shape[0]
        video_features, motion_features = [], []
        
        for i in range(batch_size):
            sl_v = seq_lens_video[i].item()
            sl_m = seq_lens_motion[i].item()
            
            v_valid, m_valid, v_pad, m_pad = torch.split(
                x[i:i+1], 
                [sl_v, sl_m, seq_len_video-sl_v, seq_len_motion-sl_m], 
                dim=1
            )
            
            video_features.append(torch.cat([v_valid, v_pad], dim=1))
            motion_features.append(torch.cat([m_valid, m_pad], dim=1))
            
        return (
            torch.cat(video_features, dim=0),
            torch.cat(motion_features, dim=0)
        )


    def _process_qkv(self, video_features, motion_features):
        """process qkv projection"""
        # video qkv
        q_v = self.norm_q(self.q(video_features)).view(video_features.shape[0], -1, self.num_heads, self.head_dim)
        k_v = self.norm_k(self.k(video_features)).view_as(q_v)
        v_v = self.v(video_features).view_as(q_v)

        # motion qkv
        q_m = self.motion_norm_q(self.motion_q(motion_features)).view(motion_features.shape[0], -1, self.num_heads, self.head_dim)
        k_m = self.motion_norm_k(self.motion_k(motion_features)).view_as(q_m)
        v_m = self.motion_v(motion_features).view_as(q_m)

        return q_v, k_v, v_v, q_m, k_m, v_m

    def _apply_position_encoding(self, q_v, k_v, q_m, k_m, seq_lens_video, seq_lens_motion, 
                                grid_sizes_video, grid_sizes_motion, freqs, freqs_motion, 
                                seq_len_video, seq_len_motion):
        """
        Applies positional encoding (RoPE) to valid parts of Q and K, handling empty sequences.
        """
        batch_size = q_v.shape[0]
        q_final_list, k_final_list = [], [] # Use _final_list to avoid confusion with q_list, k_list in RoPE

        for i in range(batch_size):
            sl_v = seq_lens_video[i].item()
            sl_m = seq_lens_motion[i].item()
            
            # Extract current sample's full (fixed-length) Q/K tensors
            q_v_i = q_v[i:i+1] # (1, seq_len_video, num_heads, head_dim)
            k_v_i = k_v[i:i+1]
            q_m_i = q_m[i:i+1] # (1, seq_len_motion, num_heads, head_dim)
            k_m_i = k_m[i:i+1]

            # Apply RoPE to video parts
            if sl_v > 0:
                q_v_valid = rope_apply(q_v_i[:, :sl_v], grid_sizes_video[i:i+1], freqs)
                k_v_valid = rope_apply(k_v_i[:, :sl_v], grid_sizes_video[i:i+1], freqs)
                q_v_pad = q_v_i[:, sl_v:seq_len_video]
                k_v_pad = k_v_i[:, sl_v:seq_len_video]
            else: # Video sequence is empty, valid parts are empty, pad parts are the whole sequence
                q_v_valid = torch.empty(1, 0, self.num_heads, self.head_dim, device=q_v.device, dtype=q_v.dtype)
                k_v_valid = torch.empty(1, 0, self.num_heads, self.head_dim, device=k_v.device, dtype=k_v.dtype)
                q_v_pad = q_v_i # Entire fixed-length tensor is padding
                k_v_pad = k_v_i
            
            # Apply RoPE to motion parts
            if sl_m > 0:
                q_m_valid = rope_apply_motion(q_m_i[:, :sl_m], grid_sizes_motion[i:i+1], freqs_motion)
                k_m_valid = rope_apply_motion(k_m_i[:, :sl_m], grid_sizes_motion[i:i+1], freqs_motion)
                q_m_pad = q_m_i[:, sl_m:seq_len_motion]
                k_m_pad = k_m_i[:, sl_m:seq_len_motion]
            else: # Motion sequence is empty
                q_m_valid = torch.empty(1, 0, self.num_heads, self.head_dim, device=q_m.device, dtype=q_m.dtype)
                k_m_valid = torch.empty(1, 0, self.num_heads, self.head_dim, device=k_m.device, dtype=k_m.dtype)
                q_m_pad = q_m_i
                k_m_pad = k_m_i
            
            # Concatenate all parts: (video_valid, motion_valid, video_pad, motion_pad)
            q_final_list.append(torch.cat([q_v_valid, q_m_valid, q_v_pad, q_m_pad], dim=1))
            k_final_list.append(torch.cat([k_v_valid, k_m_valid, k_v_pad, k_m_pad], dim=1))
            
        return torch.cat(q_final_list, dim=0), torch.cat(k_final_list, dim=0)


    def forward(self, x, seq_lens_video, seq_lens_motion, grid_sizes_video, grid_sizes_motion, 
                freqs, freqs_motion, seq_len_motion, seq_len_video, forward_type='video_with_motion'):
        return self.forward_video_with_motion(x, seq_lens_video, seq_lens_motion, grid_sizes_video, grid_sizes_motion, 
            freqs, freqs_motion, seq_len_motion, seq_len_video)
    def forward_video_with_motion(self, x, seq_lens_video, seq_lens_motion, grid_sizes_video, grid_sizes_motion, 
                freqs, freqs_motion, seq_len_motion, seq_len_video):
        batch_size = x.shape[0]
        
        # 1. Detach motion and video features
        video_features, motion_features = self._split_video_motion(
            x, seq_lens_video, seq_lens_motion, seq_len_video, seq_len_motion
        )
        
        # 2. Process Q,K,V
        q_v, k_v, v_v, q_m, k_m, v_m = self._process_qkv(video_features, motion_features)
        
        # 3. apply position encoding
        q, k = self._apply_position_encoding(
            q_v, k_v, q_m, k_m,
            seq_lens_video, seq_lens_motion,
            grid_sizes_video, grid_sizes_motion,
            freqs, freqs_motion,
            seq_len_video, seq_len_motion
        )
        
        # 4. concat value according to video_valid, motion_valid, video_pad, motion_pad
        v_list = []
        for i in range(batch_size):
            sl_v = seq_lens_video[i].item()
            sl_m = seq_lens_motion[i].item()
            
            v_v_valid, v_v_pad = torch.split(v_v[i:i+1], [sl_v, seq_len_video-sl_v], dim=1)
            v_m_valid, v_m_pad = torch.split(v_m[i:i+1], [sl_m, seq_len_motion-sl_m], dim=1)
            
            v_list.append(torch.cat([v_v_valid, v_m_valid, v_v_pad, v_m_pad], dim=1))
            
        v = torch.cat(v_list, dim=0)
        
        # 5. attn
        attn_out = flash_attention(
            q=q, k=k, v=v,
            k_lens=seq_lens_video + seq_lens_motion,
            window_size=self.window_size
        )
        
        # 6. split output and project
        video_out, motion_out = self._split_video_motion(
            attn_out.flatten(2),
            seq_lens_video, seq_lens_motion,
            seq_len_video, seq_len_motion
        )
        
        video_out = self.o(video_out)
        motion_out = self.motion_o(motion_out)
        
        # 7. concat final output
        final_output = []
        for i in range(batch_size):
            sl_v = seq_lens_video[i].item()
            sl_m = seq_lens_motion[i].item()
            
            v_valid, v_pad = torch.split(video_out[i:i+1], [sl_v, seq_len_video-sl_v], dim=1)
            m_valid, m_pad = torch.split(motion_out[i:i+1], [sl_m, seq_len_motion-sl_m], dim=1)
            
            final_output.append(torch.cat([v_valid, m_valid, v_pad, m_pad], dim=1))
            
        return torch.cat(final_output, dim=0)

class JointCrossAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.motion_q = nn.Linear(dim, dim)
        self.motion_k = nn.Linear(dim, dim)
        self.motion_v = nn.Linear(dim, dim)
        self.motion_o = nn.Linear(dim, dim)

        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.motion_norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.motion_norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens_video, seq_lens_motion, grid_sizes_video, grid_sizes_motion, 
                freqs, freqs_motion, seq_len_motion, seq_len_video, context, context_motion, context_lens, forward_type='video_with_motion'):
        return self.forward_video_with_motion(x, seq_lens_video, seq_lens_motion, grid_sizes_video, grid_sizes_motion,
                                            freqs, freqs_motion, seq_len_motion, seq_len_video, context, context_motion, context_lens)

    def forward_video_with_motion(self, x, seq_lens_video, seq_lens_motion, grid_sizes_video, grid_sizes_motion,
                                freqs, freqs_motion, seq_len_motion, seq_len_video, context, context_motion, context_lens):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # detach video and motion
        x_v_list, x_m_list = [], []
        for i in range(b):
            sl_v = seq_lens_video[i].item()
            sl_m = seq_lens_motion[i].item()
            x_v_i, x_m_i, x_v_padded_i, x_m_padded_i = torch.split(x[i:i+1], [sl_v, sl_m, seq_len_video-sl_v, seq_len_motion-sl_m], dim=1)
            x_v_list.append(torch.cat([x_v_i, x_v_padded_i], dim=1))
            x_m_list.append(torch.cat([x_m_i, x_m_padded_i], dim=1))
        x_v = torch.cat(x_v_list, dim=0)
        x_m = torch.cat(x_m_list, dim=0)

        # process qkv
        q_v = self.norm_q(self.q(x_v)).view(b, -1, n, d)
        k_v = self.norm_k(self.k(context)).view(b, -1, n, d)
        v_v = self.v(context).view(b, -1, n, d)

        q_m = self.motion_norm_q(self.motion_q(x_m)).view(b, -1, n, d)
        k_m = self.motion_norm_k(self.motion_k(context_motion)).view(b, -1, n, d)
        v_m = self.motion_v(context_motion).view(b, -1, n, d)

        # conduct cross-attn
        x_v = flash_attention(q_v, k_v, v_v, k_lens=context_lens)
        x_m = flash_attention(q_m, k_m, v_m, k_lens=context_lens)

        # ffn
        x_v = x_v.flatten(2)
        x_m = x_m.flatten(2)
        x_v = self.o(x_v)
        x_m = self.motion_o(x_m)

        # merge output
        final_output = []
        for i in range(b):
            sl_v = seq_lens_video[i].item()
            sl_m = seq_lens_motion[i].item()
            v_valid = x_v[i:i+1, :sl_v]
            m_valid = x_m[i:i+1, :sl_m]
            v_pad = x_v[i:i+1, sl_v:seq_len_video]
            m_pad = x_m[i:i+1, sl_m:seq_len_motion]
            final_output.append(torch.cat([v_valid, m_valid, v_pad, m_pad], dim=1))
            
        return torch.cat(final_output, dim=0)
        

class JointAttentionBlock(nn.Module):
    
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = LayerNorm(dim, eps)
        self.motion_norm1 = LayerNorm(dim, eps)
        self.self_attn = JointSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = LayerNorm(
            dim, eps, elementwise_affine=True
        ) if cross_attn_norm else nn.Identity()
        self.motion_norm3 = LayerNorm(
            dim, eps, elementwise_affine=True
        ) if cross_attn_norm else nn.Identity()

        self.cross_attn = JointCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = LayerNorm(dim, eps)
        self.motion_norm2 = LayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        self.motion_ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
        self.modulation_motion = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(
        self,
        x,
        e,
        e_m,
        seq_lens_video,
        seq_lens_motion,
        grid_sizes_motion,
        grid_sizes_video,
        freqs,
        freqs_motion,
        context,
        context_motion,
        context_lens,
        seq_len_motion, 
        seq_len_video,
        forward_type='video_with_motion',
        task_embedding_motion=None,
        task_embedding_video=None,
    ):
        assert e.dtype == torch.float32
        with torch.cuda.amp.autocast(dtype=torch.float32): # Assuming amp.autocast refers to torch.cuda.amp.autocast
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32
    
        assert e_m.dtype == torch.float32
        with torch.cuda.amp.autocast(dtype=torch.float32):
            e_m = (self.modulation_motion + e_m).chunk(6, dim=1)
        assert e_m[0].dtype == torch.float32

        bsz = x.shape[0]

        def detach_seq(x):
            video_list, motion_list = [], []
            for i in range(bsz):
                sl_v = seq_lens_video[i].item()
                sl_m = seq_lens_motion[i].item()

                # Ensure non-negative padding lengths for torch.split
                v_pad_len = max(0, seq_len_video - sl_v)
                m_pad_len = max(0, seq_len_motion - sl_m)

                split_sizes = [sl_v, sl_m, v_pad_len, m_pad_len]
                
                v_valid, m_valid, v_pad, m_pad = torch.split(
                    x[i:i+1], 
                    split_sizes,
                    dim=1
                )

                v_cur = torch.cat([v_valid, v_pad], dim=1)
                m_cur = torch.cat([m_valid, m_pad], dim=1)
                video_list.append(v_cur)
                motion_list.append(m_cur)
            x_video, x_motion = torch.cat(video_list, dim=0), torch.cat(motion_list, dim=0)

            return x_video, x_motion
        
        def concat_seq(x_video, x_motion):
            x_list = []
            for i in range(bsz):
                sl_v = seq_lens_video[i].item()
                sl_m = seq_lens_motion[i].item()

                v_pad_len = max(0, seq_len_video - sl_v)
                m_pad_len = max(0, seq_len_motion - sl_m)

                x_v, x_v_pad = torch.split(x_video[i:i+1], [sl_v, v_pad_len], dim=1)
                x_m, x_m_pad = torch.split(x_motion[i:i+1], [sl_m, m_pad_len], dim=1)
                
                x_list.append(torch.cat([x_v, x_m, x_v_pad, x_m_pad], dim=1))
            return torch.cat(x_list, dim=0)


        x_video, x_motion = detach_seq(x) # x_video: [B, seq_len_video, D], x_motion: [B, seq_len_motion, D]

        x_video_normalized = self.norm1(x_video).float() * (1 + e[1]) + e[0] if x_video.size(1) > 0 else torch.empty_like(x_video, dtype=torch.float32)
        x_motion_normalized = self.motion_norm1(x_motion).float() * (1 + e_m[1]) + e_m[0] if x_motion.size(1) > 0 else torch.empty_like(x_motion, dtype=torch.float32)
        
        if task_embedding_video is not None and x_video_normalized.size(1) > 0:
            if task_embedding_video.dim() == 2: # [B, D] -> [B, 1, D] -> [B, seq_len_video, D]
                task_embedding_video_repeated = task_embedding_video.unsqueeze(1).repeat(1, seq_len_video, 1).type_as(x_video_normalized)
            elif task_embedding_video.dim() == 3 and task_embedding_video.shape[1] == 1: # [B, 1, D] -> [B, seq_len_video, D]
                task_embedding_video_repeated = task_embedding_video.repeat(1, seq_len_video, 1).type_as(x_video_normalized)
            else:
                raise ValueError(f"task_embedding_video has unsupported shape: {task_embedding_video.shape}. Expected [B,D] or [B,1,D].")
            x_video_normalized = x_video_normalized + task_embedding_video_repeated

        if task_embedding_motion is not None and x_motion_normalized.size(1) > 0:
            if task_embedding_motion.dim() == 2: # [B, D] -> [B, 1, D] -> [B, seq_len_motion, D]
                task_embedding_motion_repeated = task_embedding_motion.unsqueeze(1).repeat(1, seq_len_motion, 1).type_as(x_motion_normalized)
            elif task_embedding_motion.dim() == 3 and task_embedding_motion.shape[1] == 1: # [B, 1, D] -> [B, seq_len_motion, D]
                task_embedding_motion_repeated = task_embedding_motion.repeat(1, seq_len_motion, 1).type_as(x_motion_normalized)
            else:
                raise ValueError(f"task_embedding_motion has unsupported shape: {task_embedding_motion.shape}. Expected [B,D] or [B,1,D].")
            x_motion_normalized = x_motion_normalized + task_embedding_motion_repeated
        
        # Concatenate for self-attention
        x_normalized = concat_seq(x_video_normalized, x_motion_normalized)

        # Self-Attention Calculation
        y = self.self_attn(
            x_normalized, seq_lens_video, seq_lens_motion, grid_sizes_video, grid_sizes_motion, 
            freqs, freqs_motion, seq_len_motion, seq_len_video, forward_type
        )

        y_video, y_motion = detach_seq(y)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            y_video = y_video * e[2] if y_video.size(1) > 0 else torch.empty_like(y_video)
            y_motion = y_motion * e_m[2] if y_motion.size(1) > 0 else torch.empty_like(y_motion)
            y = concat_seq(y_video, y_motion)
            if y.size(1) > 0:
                x = x + y

        # --- Cross-Attention Block Input Processing ---
        x_video, x_motion = detach_seq(x)
        x_video_normalized = self.norm3(x_video) if x_video.size(1) > 0 else torch.empty_like(x_video)
        x_motion_normalized = self.motion_norm3(x_motion) if x_motion.size(1) > 0 else torch.empty_like(x_motion)
        
        # Add Task Embeddings to x_video_normalized and x_motion_normalized
        if task_embedding_video is not None and x_video_normalized.size(1) > 0:
            task_embedding_video_repeated = task_embedding_video.unsqueeze(1).repeat(1, seq_len_video, 1).type_as(x_video_normalized) \
                                            if task_embedding_video.dim() == 2 else \
                                            task_embedding_video.repeat(1, seq_len_video, 1).type_as(x_video_normalized)
            x_video_normalized = x_video_normalized + task_embedding_video_repeated

        if task_embedding_motion is not None and x_motion_normalized.size(1) > 0:
            task_embedding_motion_repeated = task_embedding_motion.unsqueeze(1).repeat(1, seq_len_motion, 1).type_as(x_motion_normalized) \
                                            if task_embedding_motion.dim() == 2 else \
                                            task_embedding_motion.repeat(1, seq_len_motion, 1).type_as(x_motion_normalized)
            x_motion_normalized = x_motion_normalized + task_embedding_motion_repeated
        
        # Concatenate for cross-attention
        x_normalized = concat_seq(x_video_normalized, x_motion_normalized)

        # Cross-Attention Calculation
        x_cross_attn_output = self.cross_attn(x_normalized, seq_lens_video, seq_lens_motion, grid_sizes_video, grid_sizes_motion,
                                freqs, freqs_motion, seq_len_motion, seq_len_video, context, context_motion, context_lens, forward_type)
        if x_cross_attn_output.size(1) > 0:
            x = x + x_cross_attn_output
        
        # --- FFN Block Input Processing ---
        x_video, x_motion = detach_seq(x)
        x_video_normalized = self.norm2(x_video).float() * (1 + e[4]) + e[3] if x_video.size(1) > 0 else torch.empty_like(x_video, dtype=torch.float32)
        x_motion_normalized = self.motion_norm2(x_motion).float() * (1 + e_m[4]) + e_m[3] if x_motion.size(1) > 0 else torch.empty_like(x_motion, dtype=torch.float32)
        
        # Add Task Embeddings to x_video_normalized and x_motion_normalized
        if task_embedding_video is not None and x_video_normalized.size(1) > 0:
            task_embedding_video_repeated = task_embedding_video.unsqueeze(1).repeat(1, seq_len_video, 1).type_as(x_video_normalized) \
                                            if task_embedding_video.dim() == 2 else \
                                            task_embedding_video.repeat(1, seq_len_video, 1).type_as(x_video_normalized)
            x_video_normalized = x_video_normalized + task_embedding_video_repeated

        if task_embedding_motion is not None and x_motion_normalized.size(1) > 0:
            task_embedding_motion_repeated = task_embedding_motion.unsqueeze(1).repeat(1, seq_len_motion, 1).type_as(x_motion_normalized) \
                                            if task_embedding_motion.dim() == 2 else \
                                            task_embedding_motion.repeat(1, seq_len_motion, 1).type_as(x_motion_normalized)
            x_motion_normalized = x_motion_normalized + task_embedding_motion_repeated

        # FFN Calculation
        y_video = self.ffn(x_video_normalized) * e[5] if x_video_normalized.size(1) > 0 else torch.empty_like(x_video_normalized)
        y_motion = self.motion_ffn(x_motion_normalized) * e_m[5] if x_motion_normalized.size(1) > 0 else torch.empty_like(x_motion_normalized)
        
        y = concat_seq(y_video, y_motion)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            if y.size(1) > 0:
                x = x + y

        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = LayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)
    
    def forward(self, x, e):
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.main_path(x))

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

    
class MotionEncoder(nn.Module):
    def __init__(self, motion_dim=204, latent_dim=1536, global_orient_dim=6, body_pose_dim=126, 
                 transl_dim=3, joint_dim=66, transl_vel_dim=3, use_shape=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.use_shape = use_shape
        
        self.rotation_encoder = MLP(input_dim=6, output_dim=latent_dim)
        self.position_encoder = MLP(input_dim=3, output_dim=latent_dim)

        if use_shape:
            self.shape_encoder = MLP(input_dim=10, output_dim=latent_dim)
        
        # Build input structure
        structure = []
        idx = 0
        structure.append((idx, idx + global_orient_dim, "rotation"))
        idx += global_orient_dim
        for _ in range(body_pose_dim // 6):
            structure.append((idx, idx + 6, "rotation"))
            idx += 6
        structure.append((idx, idx + transl_dim, "position"))
        idx += transl_dim
        structure.append((idx, idx + transl_vel_dim, "position"))
        idx += transl_vel_dim
        for _ in range(joint_dim // 3):
            structure.append((idx, idx + 3, "position"))
            idx += 3
        if use_shape:
            structure.append((idx, idx + 10, "shape"))
            idx += 10
        
        self.input_structure = structure
        self.expected_motion_dim = idx
        self.num_tokens = len(structure)
        
        # Build indices for efficient forward pass
        rot_in, rot_out, pos_in, pos_out, shp_in, shp_out = [], [], [], [], [], []
        for i, (start, end, type_name) in enumerate(structure):
            if type_name == "rotation":
                rot_in.extend(range(start, end))
                rot_out.append(i)
            elif type_name == "position":
                pos_in.extend(range(start, end))
                pos_out.append(i)
            elif type_name == "shape":
                shp_in.extend(range(start, end))
                shp_out.append(i)
        
        self.register_buffer('rotation_indices_in', torch.LongTensor(rot_in), persistent=False)
        self.register_buffer('position_indices_in', torch.LongTensor(pos_in), persistent=False)
        self.register_buffer('rotation_indices_out', torch.LongTensor(rot_out), persistent=False)
        self.register_buffer('position_indices_out', torch.LongTensor(pos_out), persistent=False)
        if use_shape:
            self.register_buffer('shape_indices_in', torch.LongTensor(shp_in), persistent=False)
            self.register_buffer('shape_indices_out', torch.LongTensor(shp_out), persistent=False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.rotation_encoder = self.rotation_encoder.to(*args, **kwargs)
        self.position_encoder = self.position_encoder.to(*args, **kwargs)
        if self.use_shape:
            self.shape_encoder = self.shape_encoder.to(*args, **kwargs)
        self.rotation_indices_in = self.rotation_indices_in.to(*args, **kwargs)
        self.position_indices_in = self.position_indices_in.to(*args, **kwargs)
        self.rotation_indices_out = self.rotation_indices_out.to(*args, **kwargs)
        self.position_indices_out = self.position_indices_out.to(*args, **kwargs)
        if self.use_shape:
            self.shape_indices_in = self.shape_indices_in.to(*args, **kwargs)
            self.shape_indices_out = self.shape_indices_out.to(*args, **kwargs)

    def forward(self, motion_sequence):
        is_2d = motion_sequence.ndim == 2
        if is_2d:
            motion_sequence = motion_sequence.unsqueeze(0)
        
        B, T, D = motion_sequence.shape
        assert D == self.expected_motion_dim, f"Expected dim {self.expected_motion_dim}, got {D}"
        
        rotation_indices_in = self.rotation_indices_in
        position_indices_in = self.position_indices_in
        rotation_indices_out = self.rotation_indices_out
        position_indices_out = self.position_indices_out
        if self.use_shape:
            shape_indices_in = self.shape_indices_in
            shape_indices_out = self.shape_indices_out
        
        # Process rotations
        rot_data = motion_sequence[:, :, rotation_indices_in].reshape(-1, 6)
        rot_encoded = self.rotation_encoder(rot_data).view(B, T, -1, self.latent_dim)
        
        # Initialize output
        output = torch.empty(B, T, self.num_tokens, self.latent_dim, 
                           device=motion_sequence.device, dtype=rot_encoded.dtype)
        output[:, :, rotation_indices_out] = rot_encoded
        
        # Process positions
        pos_data = motion_sequence[:, :, position_indices_in].reshape(-1, 3)
        pos_encoded = self.position_encoder(pos_data).view(B, T, -1, self.latent_dim)
        output[:, :, position_indices_out] = pos_encoded
        
        # Process shapes if applicable
        if self.use_shape:
            shp_data = motion_sequence[:, :, shape_indices_in].reshape(-1, 10)
            shp_encoded = self.shape_encoder(shp_data).view(B, T, -1, self.latent_dim)
            output[:, :, shape_indices_out] = shp_encoded
        
        # Reshape to [B, latent_dim, T, num_tokens, 1]
        if is_2d:
            return output.squeeze(0).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)
        return output.permute(0, 3, 1, 2).unsqueeze(-1)

class EchoMotion_DiT(nn.Module):
    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        num_grad_checkpoints=10,
        inject_motion_type=None,
        motion_dim=201,
        motion_patch_size = (1, 1, 1),
        use_task_embedding=False,
        MoT_layers=None,
        use_shape=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_grad_checkpoints = num_grad_checkpoints
        self.global_orient_dim = 6
        self.body_pose_dim = (JOINT_NUM-1) * 6
        self.transl_dim = 3
        self.transl_vel_dim = 3
        self.joint_dim = JOINT_NUM * 3
        self.joint_vel_dim = 0
        self.motion_dim = motion_dim
        self.use_shape = use_shape
        self.shape_dim = 10

        if not self.use_shape:
            assert self.global_orient_dim + self.body_pose_dim + self.transl_dim + self.joint_dim + self.transl_vel_dim + self.joint_vel_dim == self.motion_dim
        else:
            assert self.global_orient_dim + self.body_pose_dim + self.transl_dim + self.joint_dim + self.transl_vel_dim + self.joint_vel_dim + self.shape_dim == self.motion_dim

        self.use_task_embedding = use_task_embedding

        self.inject_motion_type = inject_motion_type
        self.motion_patch_size = motion_patch_size


        # motion_head
        self.motion_encoder = MotionEncoder(
            motion_dim=self.motion_dim,
            latent_dim=self.dim,
            global_orient_dim=self.global_orient_dim,
            body_pose_dim=self.body_pose_dim,
            transl_dim=self.transl_dim,
            transl_vel_dim=self.transl_vel_dim,
            joint_dim=self.joint_dim,
            use_shape=self.use_shape,
            )
        
        num_body_joints = self.body_pose_dim // 6
        num_joint_pos = self.joint_dim // 3

        self.output_structure = []
        self.output_structure.append({"name": "global_orient", "dim": self.global_orient_dim, "type": "rotation"})
        self.output_structure.append({"name": "body_pose", "dim": self.body_pose_dim, "type": "rotation", "num_tokens": num_body_joints})
        self.output_structure.append({"name": "transl", "dim": self.transl_dim, "type": "position"})
        self.output_structure.append({"name": "transl_vel", "dim": self.transl_vel_dim, "type": "position"})
        self.output_structure.append({"name": "joint", "dim": self.joint_dim, "type": "position", "num_tokens": num_joint_pos})
        if self.use_shape:
            self.output_structure.append({"name": "betas", "dim": self.shape_dim, "type": "shape"})

        self.rotation_head = Head(self.dim, 6, motion_patch_size, eps)
        self.position_head = Head(self.dim, 3, motion_patch_size, eps)
        if self.use_shape:
            self.shape_head = Head(self.dim, 10, motion_patch_size, eps)

        rotation_token_indices = []
        position_token_indices = []
        shape_token_indices = []
        
        rotation_dim_indices = []
        position_dim_indices = []
        shape_dim_indices = []
        current_token_idx = 0 
        current_dim_idx = 0 

        for part_info in self.output_structure:
            num_tokens = part_info.get("num_tokens", 1)
            dim_ = part_info["dim"]
            part_type = part_info["type"]
            
            token_indices = list(range(current_token_idx, current_token_idx + num_tokens))
            dim_indices = list(range(current_dim_idx, current_dim_idx + dim_))
            
            if part_type == "rotation":
                rotation_token_indices.extend(token_indices)
                rotation_dim_indices.extend(dim_indices)
            elif part_type == "position":
                position_token_indices.extend(token_indices)
                position_dim_indices.extend(dim_indices)
            elif part_type == "shape":
                shape_token_indices.extend(token_indices)
                shape_dim_indices.extend(dim_indices)
            
            current_token_idx += num_tokens
            current_dim_idx += dim_

            self.register_buffer('rotation_token_indices_out', torch.LongTensor(rotation_token_indices), persistent=False)
            self.register_buffer('position_token_indices_out', torch.LongTensor(position_token_indices), persistent=False)
            self.register_buffer('rotation_dim_indices_out', torch.LongTensor(rotation_dim_indices), persistent=False)
            self.register_buffer('position_dim_indices_out', torch.LongTensor(position_dim_indices), persistent=False)
            if self.use_shape:
                self.register_buffer('shape_token_indices_out', torch.LongTensor(shape_token_indices), persistent=False)
                self.register_buffer('shape_dim_indices_out', torch.LongTensor(shape_dim_indices), persistent=False)
        
        # video head
        self.head = Head(dim, out_dim, patch_size, eps)


        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )

        self.text_embedding_motion = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        self.time_embedding_motion = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )

        self.time_projection_motion = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )

        if self.use_task_embedding:
            self.task_embedding_video = nn.Sequential(
                nn.Linear(2 * freq_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )
            self.task_embedding_motion = nn.Sequential(
                nn.Linear(2 * freq_dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )

        # MoT layer id
        self.MoT_layers = MoT_layers
        self.blocks = []
        for i in range(self.num_layers):
            if i in MoT_layers:
                self.blocks.append(JointAttentionBlock(
                    dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps
                ))
            else:
                self.blocks.append(AttentionBlock(
                    dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps
                ))
        self.blocks = nn.ModuleList(self.blocks)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(128, d - 4 * (d // 6), theta=10000),
            rope_params(128, 2 * (d // 6), theta=10000),
            rope_params(128, 2 * (d // 6), theta=10000)
        ], dim=1)

        self.freqs_motion = torch.cat([
            rope_params(128, d - 4 * (d // 6), theta=10000, stride=0.25, start_idx=0), # frame wise syncronized
            rope_params(128, 4 * (d // 6), theta=10000, stride=1, start_idx=128), # motion space position encoding
        ], dim=1)

        self.init_weights()

    def _make_task_encoding(self, v, m, device):
        """Create task encoding from video and motion flags."""
        emb_v = sinusoidal_embedding_1d(self.freq_dim, torch.tensor([v])).float().to(device)
        emb_m = sinusoidal_embedding_1d(self.freq_dim, torch.tensor([m])).float().to(device)
        return torch.cat([emb_v, emb_m], dim=-1)

    def forward(self, x, t, context, context_motion, seq_len_video, seq_len_motion, 
                motion, pred_task=['video_with_motion'], t_motion=None):

        device = self.text_embedding[0].weight.device
        ts_motion = t_motion if t_motion is not None else t
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
            self.freqs_motion = self.freqs_motion.to(device)

        # Process task embeddings
        task_map = {
            'video_with_motion': (1., 1., lambda i: (t[0], ts_motion[0])),
            'video_to_motion': (0., 1., lambda i: (torch.tensor(1.).to(device), ts_motion[0])),
            'motion_to_video': (1., 0., lambda i: (t[0], torch.tensor(1.).to(device)))
        }
        
        if pred_task[0] not in task_map:
            raise ValueError(f'Unknown pred_task: {pred_task[0]}')
        
        v_flag, m_flag, t_func = task_map[pred_task[0]]
        task_one_hot_embeding = self._make_task_encoding(v_flag, m_flag, t.device).unsqueeze(0)
        ts_video, ts_motion = t_func(0)
        ts_video = ts_video.unsqueeze(0) if ts_video.dim() == 0 else ts_video
        ts_motion = ts_motion.unsqueeze(0) if ts_motion.dim() == 0 else ts_motion
        
        task_embedding_motion = self.task_embedding_motion(task_one_hot_embeding).squeeze(1)
        task_embedding_video = self.task_embedding_video(task_one_hot_embeding).squeeze(1)

        # process motion and video time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, ts_video).float()
            )
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

            e_m = self.time_embedding_motion(
                sinusoidal_embedding_1d(self.freq_dim, ts_motion).float()
            )

            e_m0 = self.time_projection_motion(e_m).unflatten(1, (6, self.dim))
            assert e_m0.dtype == torch.float32 and e_m.dtype == torch.float32

        # video latents processing
        video_x = []
        seq_lens_video = []
        video_grid_sizes_list = []

        for u in x:
            if u.shape[1] > 0:
                patch_embedded_video = self.patch_embedding(u.unsqueeze(0))
                video_grid_sizes_list.append(torch.tensor(patch_embedded_video.shape[2:], dtype=torch.long, device=device))
                flattened_video = patch_embedded_video.flatten(2).transpose(1, 2) # [B, L, D]
                video_x.append(flattened_video)
                seq_lens_video.append(flattened_video.size(1))
            else:
                empty_tensor = torch.empty(1, 0, self.dim, dtype=torch.float, device=device)
                video_x.append(empty_tensor)
                seq_lens_video.append(0)
                video_grid_sizes_list.append(torch.tensor([0, 0, 0], dtype=torch.long, device=device)) 

        video_grid_sizes = torch.stack(video_grid_sizes_list) # [B, 3]
        seq_lens_video = torch.tensor(seq_lens_video, dtype=torch.long, device=device) # [B]

        assert seq_lens_video.max() <= seq_len_video

        # motion latents processing
        motion_x = []
        seq_lens_motion = []
        motion_gride_sizes_list = []

        for m in motion:
            if m.shape[0] > 0:
                motion_feat = self.motion_encoder(m)
                motion_feat = motion_feat.to(device)
                motion_gride_sizes_list.append(torch.tensor(motion_feat.shape[2:], dtype=torch.long, device=device))
                flattened_motion_feat = motion_feat.flatten(2).transpose(1, 2)
                motion_x.append(flattened_motion_feat)
                seq_lens_motion.append(flattened_motion_feat.size(1))
            else:  
                empty_tensor = torch.empty(1, 0, self.dim, dtype=torch.float, device=device)
                motion_x.append(empty_tensor)
                seq_lens_motion.append(0)
                motion_gride_sizes_list.append(torch.tensor([0,0,0], dtype=torch.long, device=device))

        motion_grid_sizes = torch.stack(motion_gride_sizes_list)
        seq_lens_motion = torch.tensor(seq_lens_motion, dtype=torch.long, device=device)

        assert seq_lens_motion.max() <= seq_len_motion

        # concat the video and motion latents
        x = []
        for u, v in zip(video_x, motion_x):
            u = torch.cat([u, v], dim=1)
            u = torch.cat([u, u.new_zeros(1, seq_len_motion+seq_len_video - u.size(1), u.size(2))],dim=1)
            x.append(u)
        x = torch.cat(x, dim=0)

        # process context
        context_lens = None
        context = self.text_embedding(torch.stack([torch.cat([
            u, u.new_zeros(self.text_len - u.size(0), u.size(1))
        ]) for u in context]))
        context_motion = self.text_embedding(torch.stack([torch.cat([
            u, u.new_zeros(self.text_len - u.size(0), u.size(1))
        ]) for u in context_motion]))

        # process arguments for MoT
        kwargs = dict(
            e=e0,
            e_m=e_m0,
            seq_lens_video=seq_lens_video,
            seq_lens_motion=seq_lens_motion,
            grid_sizes_video=video_grid_sizes, 
            grid_sizes_motion=motion_grid_sizes,
            freqs=self.freqs,
            freqs_motion=self.freqs_motion,
            context=context,
            context_motion=context_motion,
            context_lens=context_lens,
            seq_len_motion=seq_len_motion, 
            seq_len_video=seq_len_video,
            forward_type='video_with_motion',
            task_embedding_motion=task_embedding_motion,
            task_embedding_video=task_embedding_video,
        )

        for block in self.blocks:
            x = block(x, **kwargs)

        video_latent_list = []
        motion_latent_list = []

        for i in range(x.shape[0]):
            sl_v = seq_lens_video[i].item()
            sl_m = seq_lens_motion[i].item()
            split_sizes = [
                sl_v,  
                sl_m,  
                seq_len_video - sl_v,  
                seq_len_motion - sl_m  
            ]
            v_latent, m_latent, v_padded, m_padded = torch.split(x[i:i+1], split_sizes, dim=1)
            v_latent = torch.cat([v_latent, v_padded], dim=1)
            m_latent = torch.cat([m_latent, m_padded], dim=1)

            video_latent_list.append(v_latent)
            motion_latent_list.append(m_latent)

        video_latent = torch.cat(video_latent_list, dim=0)
        motion_latent = torch.cat(motion_latent_list, dim=0)

        # Decode motion latents
        video_out = self.head(video_latent, e)

        B = motion_latent.shape[0]
        num_tokens = self.motion_encoder.num_tokens
        assert seq_len_motion % num_tokens == 0, "Motion sequence length not divisible by num_tokens"
        T_max = seq_len_motion // num_tokens
        
        motion_latent_grid = motion_latent.view(B, T_max, num_tokens, self.dim)
        motion_out = torch.empty(B, T_max, self.motion_dim, 
                                device=motion_latent.device, dtype=motion_latent.dtype)

        # Decode rotations
        rot_latents = motion_latent_grid[:, :, self.rotation_token_indices_out].reshape(B, -1, self.dim)
        motion_out[:, :, self.rotation_dim_indices_out] = self.rotation_head(rot_latents, e_m).view(B, T_max, -1)

        # Decode positions
        pos_latents = motion_latent_grid[:, :, self.position_token_indices_out].reshape(B, -1, self.dim)
        motion_out[:, :, self.position_dim_indices_out] = self.position_head(pos_latents, e_m).view(B, T_max, -1)

        # Decode shapes if applicable
        if self.use_shape:
            shp_latents = motion_latent_grid[:, :, self.shape_token_indices_out].reshape(B, -1, self.dim)
            motion_out[:, :, self.shape_dim_indices_out] = self.shape_head(shp_latents, e_m).view(B, T_max, -1)

        motion_out = self.unpatchify_motion(motion_out, motion_grid_sizes[:, 0]) 
        video_out = self.unpatchify(video_out, video_grid_sizes)

        return [u.float() for u in video_out], [m.float() for m in motion_out]
    
    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def unpatchify_motion(self, x, real_lengths):
            out = []
            for i in range(x.shape[0]):
                length = real_lengths[i]
                real_motion_sequence = x[i, :length, :]
                out.append(real_motion_sequence)
            return out
    
    def init_weights(self):

        for m in self.time_embedding_motion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules(): 
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding_motion.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        if self.use_task_embedding:
            for m in self.task_embedding_motion.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
            for m in self.task_embedding_video.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
        
        nn.init.zeros_(self.head.head.weight)
        nn.init.zeros_(self.rotation_head.head.weight)
        nn.init.zeros_(self.position_head.head.weight)
        if self.use_shape:
            nn.init.zeros_(self.shape_head.head.weight)
    