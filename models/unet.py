
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # At this point we could implement on top of the sinusoide embedding a small MLP to project better

    def sinusoidal_embedding(self, t):
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) * (torch.log(torch.tensor(10000.0)) / (half - 1)))
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

    def forward(self,t):
        return self.sinusoidal_embedding(t)

class ResBlock(nn.Module):
    def __init__(self, in_channels:int,out_channels:int,emb_dim:int):
        """
        in_channels : number of input channels
        out_channels : number of output channels
        emb_dim : dimension of the timestep embedding
        """

        super().__init__()

        # First block
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels), #split channels into group of 8
            nn.SiLU(),
            nn.Conv2d(in_channels,out_channels, kernel_size=3,padding=1)
        )

        # Embedding projection
        # We need to project the embedding in order to be in the same space while adding
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

        # Second block
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels,out_channels, kernel_size=3,padding=1)
        )

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) #too match if different
        else:
            self.skip = nn.Identity()

    def forward(self,x,t_emb):
        """
        x     : (B, in_channels, H, W)
        t_emb : (B, emb_dim)
        """

        # First block
        # (B, out_channels, H, W)
        h = self.block1(x)

        # Add time step embedding
        # (B, emb_dim)
        t = self.time_proj(t_emb)
        # (B, out_channels)

        # Add by broadcasting
        h = h + t[:, :, None, None]

        # Second block
        # (B, out_channels, H, W)
        h = self.block2(h)

        # Skip connection
        return h + self.skip(x)
    
class SelfAttention(nn.Module):
    def __init__(self, channels:int,n_heads:int=4):
        """
        channels : number of input/output channels
        n_heads  : number of attention heads
        """
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads

        # Normalization
        self.norm = nn.GroupNorm(8, channels)

        # Q,K,V Linear
        self.qkv = nn.Linear(channels, channels * 3)
        self.output = nn.Linear(channels,channels)

    def forward(self,x):
        """
        x : (B, C, H, W)
        """

        B,C,H,W = x.shape

        # Norm
        h = self.norm(x)

        # Flatten
        h = h.view(B, C, H * W).transpose(1, 2) # (B, H*W, C)

        # Compute Q, K, V
        qkv = self.qkv(h) # (B, H*W, 3*C)
        q,k,v = qkv.chunk(3, dim=-1) # 3*(B, H*W, C)

        # At this point, the differents head are inside q,k,v s we need to split thoses matrix depending on n_heads
        def split_heads(t):
            return t.view(B, H * W, self.n_heads, self.head_dim).transpose(1, 2) # (B, n_heads, H*W, head_dim)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Dot-product attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_heads, H*W, H*W)
        attn = torch.softmax(scores, dim=-1) # (B, n_heads, H*W, H*W)

        # Weighted sum of values
        out = torch.matmul(attn, v) # (B, n_heads, H*W, head_dim)

        # Merge heads
        out = out.transpose(1, 2).contiguous()  # (B, H*W, n_heads, head_dim)
        out = out.view(B, H * W, C) # (B, H*W, C)

        # Output projection
        out = self.output(out) # (B, H*W, C)

        # Reshape
        out = out.transpose(1, 2).view(B, C, H, W) # (B, C, H, W)

        # Return with residual
        return out + x 
    
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads=4):
        """
        query_dim   : dimension of image features (from U-Net)
        context_dim : dimension of text embeddings (from Text Encoder)
        n_heads     : number of attention heads
        """
        super().__init__()
        assert query_dim % n_heads == 0, "query_dim must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = query_dim // n_heads
        self.scale = self.head_dim  -0.5

        # Normalization
        self.norm_image = nn.LayerNorm(query_dim)
        self.norm_text = nn.LayerNorm(context_dim)

        # Q from image, K and V from text
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(context_dim, query_dim)
        self.v_proj = nn.Linear(context_dim, query_dim)

        # Output
        self.out_proj = nn.Linear(query_dim, query_dim)

    def forward(self, x, context):
        """
        x       : (B, H*W, query_dim)   image features
        context : (B, seq_len, context_dim) text embeddings
        returns : (B, H*W, query_dim)
        """

        # Get the shape
        B, HW, _ = x.shape
        _, seq_len, _ = context.shape

        # Normalize
        x_norm = self.norm_image(x)
        ctx_norm = self.norm_text(context)

        # Compute Q, K, V 
        q = self.q_proj(x_norm) # (B, H*W, query_dim)

        # projection of context_dim to query_dim
        k = self.k_proj(ctx_norm) # (B, seq_len, query_dim)
        v = self.v_proj(ctx_norm) # (B, seq_len, query_dim)

        # Reshape
        q = q.view(B, HW,self.n_heads, self.head_dim).transpose(1, 2) # (B, n_heads, HW, head_dim)
        k = k.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # (B, n_heads, seq_len, head_dim)
        v = v.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # (B, n_heads, seq_len, head_dim)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v) # (B, n_heads, HW, head_dim)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, HW, -1) # (B, HW, query_dim)

        # Output + residual
        return self.out_proj(out) + x # (B, H*W, query_dim)
        

class UNet(nn.Module):
    def __init__(self, 
            in_channels:int=3,
            base_channels:int=32,
            channel_mults = (1,2,4),
            n_heads = 4,
            emb_dim =128,
            context_dim=128):
        super().__init__()

        self.emb_dim = emb_dim
        channels = [base_channels * i for i in channel_mults]

        # Timestep embedding
        self.time_emb = TimestepEmbedding(emb_dim)

        # init conv
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)


        ###########
        # Encoder #
        ###########

        # To increase the number of channels and learn multiple features
        self.encoder_blocks = nn.ModuleList()
        # To reduce the size of the input and see the features more precisely/largelly
        self.downsamplers = nn.ModuleList()

        in_ch = base_channels
        for out_ch in channels:
            self.encoder_blocks.append(nn.ModuleList([
                ResBlock(in_ch,  out_ch, emb_dim), # Changes the channel dimension
                ResBlock(out_ch, out_ch, emb_dim), # Work in the same space -> extracts more complex patterns
            ]))
            self.downsamplers.append(
                nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            in_ch = out_ch

        ##############
        # Bottleneck #
        ##############
        mid_ch = channels[-1]

        bottleneck_blocks = [
            ResBlock(mid_ch, mid_ch, emb_dim),
            SelfAttention(mid_ch, n_heads),
        ]

        if context_dim is not None:
            bottleneck_blocks.append(CrossAttention(mid_ch, context_dim))

        bottleneck_blocks.append(ResBlock(mid_ch, mid_ch, emb_dim))

        self.bottleneck = nn.ModuleList(bottleneck_blocks)

        ###########
        # Decoder #
        ###########
        # Reduce the number of channels and features
        self.decoder_blocks = nn.ModuleList()
        # Increase the size of the value inside the bottlenck
        self.upsamplers = nn.ModuleList()

        for out_ch in reversed(channels):
            self.upsamplers.append(
                nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1)
            )
            self.decoder_blocks.append(nn.ModuleList([
                ResBlock(in_ch + out_ch, out_ch, emb_dim),   # +out_ch for skip connection
                ResBlock(out_ch, out_ch, emb_dim),
            ]))
            in_ch = out_ch

        # Final conv that predict the noise
        self.final = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, t, text_emb=None):
        """
        x : (B, 3, 32, 32) noisy image
        t : (B,)            timestep
        text_emb : (B, seq_len, context_dim)
        returns predicted noise ε (B, 3, 32, 32)
        """

        # Timestep embedding
        t_emb = self.time_emb(t) # (B, emb_dim)

        x = self.init_conv(x) 

        ### Encoder ###
        skips = []
        for (res1, res2), down in zip(self.encoder_blocks, self.downsamplers):
            x = res1(x, t_emb) # First ResBlock
            x = res2(x, t_emb) # Second ResBlock
            skips.append(x)
            x = down(x) # DownSample

        ### Bottleneck ###
        B, C, H, W = x.shape
        
        if len(self.bottleneck) == 4:
            res1, attn, cross, res2 = self.bottleneck
        else:
            res1, attn, res2 = self.bottleneck
            cross = None

        # Classic
        x = res1(x, t_emb)
        x = attn(x)

        # If text conditionning
        if cross is not None and text_emb is not None:
            x_flat = x.view(B, C, H * W).transpose(1, 2) # (B, H*W, C)
            x_flat = cross(x_flat, text_emb) # (B, H*W, C)
            x = x_flat.transpose(1, 2).view(B, C, H, W) # (B, C, H, W)

        # Classic
        x = res2(x, t_emb)

        ### Decoder ###
        for (res1, res2), up, skip in zip(self.decoder_blocks, self.upsamplers, reversed(skips)):
            x = up(x) # UpSample
            x = torch.cat([x, skip], dim=1) # Add the residual
            x = res1(x, t_emb) # First ResBlock
            x = res2(x, t_emb) # Second ResBlock

        # Final conv
        return self.final(x)