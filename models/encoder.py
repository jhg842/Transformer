import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        bat, seq_len, d_model = x.shape
        
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        q = q.view(bat, -1, self.n_heads, self.d_model//self.n_heads).transpose(1,2)
        k = k.view(bat, -1, self.n_heads, self.d_model//self.n_heads).transpose(1,2)
        v = v.view(bat, -1, self.n_heads, self.d_model//self.n_heads).transpose(1,2)
        
        att_weights = (q @ k.transpose(-2,-1)) / (self.d_model**0.5)
        att_score = F.softmax(att_weights, dim=-1)
        
        attn_output = att_score @ v
        
        attn_output = attn_output.transpose(1,2).contiguous().view(bat, seq_len, self.d_model)
        
        return attn_output
    
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, feed_dim):
        super().__init__()
        
        self.attn = Encoder(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, feed_dim),
            nn.GELU(),
            nn.Linear(feed_dim, d_model)
        )
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        
    def forward(self, x):
        x1 = self.norm1(x)
        x1 = self.attn(x1)
        x1 = self.dropout1(x1) + x
        
        x2 = self.norm2(x1)
        x2 = self.ffn(x2)
        x2 = self.dropout2(x2) + x1
        
        return x2
    
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, d_model, n_heads, feed_dim, n_layers):
        super().__init__()
        
        self.layers = _get_clones(encoder_layer, n_layers)
        self.n_layers = n_layers
        
    def forward(self, x):
        for layer in self.layers:
            output = layer(x)
            
        return output
        
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    

