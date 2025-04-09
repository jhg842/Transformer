import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import TransformerEncoder, EncoderLayer
from decoder import TransformerDecoder, DecoderLayer
from position_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, n_enc_layers, n_dec_layers,
                 feed_dim):
        super().__init__()
        
        self.position_encoding = PositionalEncoding(d_model, max_length = 512, temp = 10000)
        encoder_layer = EncoderLayer(d_model, n_heads, feed_dim)
        decoder_layer = DecoderLayer(d_model, n_heads, feed_dim)
        self.encoder = TransformerEncoder(encoder_layer, d_model, n_heads, feed_dim, n_enc_layers)
        self.decoder = TransformerDecoder(decoder_layer, d_model, n_heads, feed_dim, n_dec_layers)
    
    def forward(self, enc_input, dec_input, mask = None):
        enc_input = self.position_encoding(enc_input)
        memory = self.encoder(enc_input)
        
        output = self.decoder(dec_input, memory, mask)
        
        return output
    

a = torch.randn(1,50,256)
b = torch.randn(1,50,256)
model = Transformer(d_model = 256, n_heads = 8, n_enc_layers = 6, n_dec_layers = 6, feed_dim = 256)
print(model(a,b,None).shape)

def build_transformer(args):
    
    return Transformer(
        d_model = args.d_model,
        n_heads = args.n_heads,
        n_enc_layers = args.n_enc_layers,
        n_dec_layers = args.n_dec_layers,
        feed_dim = args.feed_dim
    )