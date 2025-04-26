import torch
import torch.nn as nn

import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken



## Chapter 02 ##
class GPTDatasetV1(Dataset):

    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


## Chapter 03 ##
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_lenth, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_lenth, context_lenth), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Splitting Bigger Matrix into indidual heads
        ## (batch, num_tokens(seq_length), num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transposes From
        ##    0         1           2         3
        ### (batch, num_tokens, num_heads, head_dim)
        ## To (1, 2) dimensions are interchanged
        ##  (batch, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        ## key dimension after transpose (batch, num_heads, head_dim, num_tokens) (batch, num_heads, num_tokens, head_dim)
        ## atten_score dim (batch, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  ##

        ## (batch, num_heads, num_tokens, head_dim) -> transpose(1,2) -> (batch, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        context_vec = self.out_proj(context_vec)
        return context_vec
    
## Chapter 04 ##

class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):

        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0) / torch.tensor(torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)  
    
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # prevents division by zero error
        self.scale = nn.Parameter(
            torch.ones(emb_dim)
        )  # Default Scale & Shift is set to 1 & 0
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # But network can learn them

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        out = norm_x * self.scale + self.shift
        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.attn_block = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_lenth=config["context_length"],
            num_heads=config["n_heads"],
            dropout=config["drop_rate"],
            qkv_bias=config["qkv_bias"],
        )
        self.feedforward_block = FeedForward(config)
        self.layer_norm_1 = LayerNorm(config["emb_dim"])
        self.layer_norm_2 = LayerNorm(config["emb_dim"])
        self.droput = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        # Block MHA
        ## Identity for creating skip connection
        x_identity = x
        ## Layer Norm on Input
        x_norm_trans = self.layer_norm_1(x)
        ## MHA
        attn_output = self.attn_block(x_norm_trans)
        ## DropOut
        attn_dropout = self.droput(attn_output)
        ## Adding identity to complete skip connection
        x = attn_dropout + x_identity

        # Block FFCN
        ## Identity for creating skip connection
        x_identity = x
        ## Layer Norm on Input
        x_norm_trans = self.layer_norm_2(x)
        ## FFCN Output
        ffn_output = self.feedforward_block(x_norm_trans)
        ## Dropout
        ffn_dropput = self.droput(ffn_output)

        x = ffn_dropput + x_identity
        return x


class GPTModel(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.size()
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        ## Dropout before transformer blocks
        x = self.drop(x)
        x = self.trf_blocks(x)
        ## layer norm after transformer blocks
        x = self.final_norm(x)
        ## Embedding space to vocab space
        logits = self.out_head(x)
        return logits