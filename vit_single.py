import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ViTSingle(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, batch=16, is_sinusoid=True, is_first=False, pool='mean',
                 dim_head=64, dropout=0., emb_dropout=0., embedding_cnn=True):
        super().__init__()
        # assert image_size_w % patch_size == 0 or image_size_h % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # num_patches = (image_size_h // patch_size) * (image_size_w // patch_size)
        # patch_dim = channels * patch_size ** 2
        self.to_patch_embedding_cnn = embedding_cnn
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # if not embedding_cnn:
        #     self.to_patch_embedding = nn.Sequential(
        #         Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        #         nn.Linear(patch_dim, dim),
        #     )
        # else:
        #     self.to_patch_embedding_cnn = embedding_cnn
        self.is_first = is_first
        self.is_sin = is_sinusoid
        if self.is_sin:
            if self.is_first:
                self.pos_embedding = nn.Parameter(torch.randn(1, 144, 128))
            else:
                self.pos_embedding = nn.Parameter(torch.randn(1, 8, 128))
        else:
            if self.is_first:
                self.pos_embedding = self.get_sinusoid_table(batch, 144, 128)
            else:
                self.pos_embedding = self.get_sinusoid_table(batch, 8, 128)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        # self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128)
        )

    def forward(self, img, mask=None):
        # self.x = img
        # is_sinusoid = True
        batch = img.shape[0]
        pos_is_add = True
        if self.is_first:
            if self.to_patch_embedding_cnn:
                x, shape = self.embedding_cnn(img)
                # b, n, d_model = shape
                if pos_is_add:
                    x += self.pos_embedding.repeat([batch, 1, 1])
                else:
                    x = torch.mul(x, self.pos_embedding.repeat([batch, 1, 1]))
            else:
                x = self.to_patch_embedding(img)
                # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
                # x = torch.cat((cls_tokens, x), dim=1)
                if pos_is_add:
                    x = x + self.pos_embedding
                else:
                    x = torch.mul(x, self.pos_embedding.repeat([batch, 1, 1]))
            x = self.dropout(x)

            x = self.transformer(x, mask)
        else:
            # x, shape = self.embedding_cnn(img)
            # x = img + self.pos_embedding
            if pos_is_add:
                x = img + self.pos_embedding.repeat([batch, 1, 1])
            else:
                x = torch.mul(img, self.pos_embedding.repeat([batch, 1, 1]))
            x = self.dropout(x)

            x = self.transformer(x, mask)
            """
            # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            # x = self.mlp_head(x)
            not simclr
            """

            x = x.view(x.shape[0], -1)

        # x = self.to_latent(x)
        return x

    def embedding_cnn(self, x, projection_num=128):
        b, c, h, w = x.shape
        x = nn.Conv2d(4, 32, kernel_size=10, stride=10).to(x.device)(x)
        x = nn.Conv2d(32, projection_num, kernel_size=3, stride=3).to(x.device)(x)
        x = x.view(b, projection_num, -1).permute(0, 2, 1)
        return x, x.shape

    def pos_embed(self, seq_len, d_model):
        pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        return pos_embedding


    def get_sinusoid_table(self,b, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i // 2)) / d_model)

        sinusoid_table = np.zeros((b, seq_len, d_model))
        # for batch in range(b):
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

        return torch.from_numpy(sinusoid_table).float()