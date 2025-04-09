import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Decoder(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)  
        self.v = nn.Linear(d_model, d_model)
        
    def forward(self, qx,kx,vx, mask = None):
        bat, seq_len, d_model = qx.shape
        q = self.q(qx)
        k = self.k(kx)
        v = self.v(vx)
        q = q.view(bat, -1 ,self.n_heads, self.d_model // self.n_heads).transpose(1,2)
        k = k.view(bat, -1 ,self.n_heads, self.d_model // self.n_heads).transpose(1,2)
        v = v.view(bat, -1 ,self.n_heads, self.d_model // self.n_heads).transpose(1,2)
        
        attn_weight = (q @ k.transpose(-2,-1)) / (self.d_model ** 0.5)
        
        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask == 0, float('-inf'))
        
        attn_score = F.softmax(attn_weight, dim=-1)
        attn_output = attn_score @ v
        attn_output = attn_output.transpose(1,2).contiguous().view(bat, seq_len, self.d_model)
        
        return attn_output
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, feed_dim):
        super().__init__()
        
        self.masked_attn = Decoder(d_model, n_heads)
        self.cross_attn = Decoder(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, feed_dim),
            nn.GELU(),
            nn.Linear(feed_dim, d_model)
        )
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        
    def forward(self, x, memory, mask = None):
        x1 = self.norm1(x)
        x1 = self.masked_attn(x1,x1,x1, mask)
        x1 = self.dropout1(x1) + x
        
        x2 = self.norm2(x1)
        x2 = self.cross_attn(x2,memory,memory, mask)
        x2 = self.dropout2(x2) + x1
        
        x3 = self.norm3(x2)
        x2 = self.ffn(x3)
        
        return x3
    
    
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, d_model, n_heads, n_layers, feed_dim):
        super().__init__()
        
        self.layers = _get_clones(decoder_layer, n_layers)
        self.n_layers = n_layers
        
    def forward(self, x, memory, mask = None):
        
        for layer in self.layers:
            output = layer(x, memory, mask)
            
        return output


def _get_clones(module, N):
    
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


