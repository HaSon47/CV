import torch
from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor
from einops import repeat

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 3, patch_size = 8, emb_size = 128):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout)
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.att(x,x,x)
        return attn_output
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
class PositionalEncoding(nn.Module):

    def __init__(self, emb_dim, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.emb_dim = emb_dim

    def forward(self):
        even_i = torch.arange(0, self.emb_dim, 2).float()
        denominator = torch.pow(10000, even_i/self.emb_dim)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE
    
class ViT(nn.Module):
    def __init__(self, config, device='cpu'):
        super(ViT, self).__init__()

        #Attributes
        self.channels = config.ViT.channels
        self.height = config.ViT.img_size
        self.width = config.ViT.img_size
        self.patch_size = config.ViT.patch_size
        self.n_layers = config.ViT.n_layers
        self.emb_dim = config.ViT.emb_dim
        self.dropout = config.ViT.dropout
        self.out_dim = config.ViT.out_dim
        self.device = device
        self.heads = config.ViT.heads

        #Patching
        self.patch_embedding = PatchEmbedding(in_channels=self.channels,
                                              patch_size=self.patch_size,
                                              emb_size= config.ViT.emb_dim)
        
        #Learnable params
        num_patches = (self.height //self.patch_size)**2
        self.pos_embedding= PositionalEncoding(self.emb_dim, num_patches+1).forward(). view(1,num_patches+1,self.emb_dim ).to(self.device)
        self.cls_token = nn.Parameter(torch.rand(1,1, self.emb_dim))

        #Transformer Encoder
        self.layers = nn.ModuleList([])
        for _ in range(self.n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(self.emb_dim, Attention(self.emb_dim, n_heads = self.heads, dropout = self.dropout))),
                ResidualAdd(PreNorm(self.emb_dim,  FeedForward(self.emb_dim, self.emb_dim, dropout = self.dropout)))
            )
            self.layers.append(transformer_block)

        # Classificaition head
        self.head = nn.Sequential(nn.LayerNorm(self.emb_dim), nn.Linear(self.emb_dim, self.out_dim))

    def forward(self, img):
        # Get patch embedding vectors
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        # Add cls token to inputs
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :(n+1)]

        # Transformer layers
        for i in range(self.n_layers):
            x = self.layers[i](x)

        # Output based on classification token
        return self.head(x[:, 0, :])

    