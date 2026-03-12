import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=4, ff_mult=4):
        super().__init__()

        # Self Attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

        # MLP/Feed Forward
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim)
        )

    def forward(self, x, padding_mask=None):
        
        # Attention
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=padding_mask)
        # Residual
        x  = x + attn_out

        # Feed Forward + Residual
        x = x + self.ff(self.norm2(x))
        return x
    

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, dim=128, n_layers=4, n_heads=4, max_len=28):
        """
        vocab_size : size of your tokenizer vocabulary
        dim        : embedding dimension
        n_layers   : number of transformer blocks
        n_heads    : number of attention heads
        max_len    : maximum sequence length
        """
        super().__init__()

        self.dim = dim

        # Embedding (Pos + Token)
        # Each token is projected into a space of dimension dim
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        # Reguarding the position of the token he is also projected into a space of dimension dim
        self.pos_emb = nn.Embedding(max_len, dim)

        # Transformersblocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads) for _ in range(n_layers)
        ])

        # Norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, padding_mask=None):
        """
        tokens       : (B, seq_len) token ids
        padding_mask : (B, seq_len) True where padding
        returns      : (B, seq_len, dim) text embeddings
        """
        B, seq_len = tokens.shape

        # Embedding
        # To get the positions of each tokens
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)  # (1, seq_len)

        # Compute the embedding
        x = self.token_emb(tokens) + self.pos_emb(positions) # (B, seq_len, dim)

        # Transformers block
        for block in self.blocks:
            x = block(x, padding_mask) # to not influence attetion 

        return self.norm(x)   # (B, seq_len, dim)