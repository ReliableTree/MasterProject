import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from pathlib import Path

class TransformerModel(nn.Module):

    def __init__(self, model_setup = None, dropout=0.2):
        super().__init__()
        self.model_type = 'Transformer'
        if model_setup is not None:
            d_model = model_setup['d_model']
            nhead = model_setup['nhead']
            d_hid = model_setup['d_hid']
            d_inpt = model_setup['d_inpt']
            nlayers = model_setup['nlayers']
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=nlayers)
        self.encoder = nn.Linear(d_inpt, d_model)
        self.d_model = d_model


    def forward(self, src: Tensor, src_mask = None) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = src.transpose(0,1)
        src = self.encoder(src)* math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = output.transpose(0,1)

        return output

class TransformerDecoder(nn.Module):
    def __init__(self, model_setup) -> None:
        super().__init__()
        self.d_output = model_setup['d_output']
        self.super_init = False
        self.output_seq = model_setup['output_seq']

    def forward(self, inpt):
        #inpt N,S,D
        if not self.super_init:
            if self.output_seq:
                self.decoder = nn.Linear(inpt.size(-1), self.d_output)
            else:
                self.decoder = nn.Linear(inpt.size(-1) * inpt.size(-2), self.d_output)
            self.decoder.to(inpt.device)
            self.super_init = True

        if not self.output_seq:
            inpt = inpt.reshape(inpt.size(0), -1)
        output = self.decoder(inpt)
        
        return output

class TransformerDecoder2(nn.Module):
    def __init__(self, model_setup) -> None:
        super().__init__()
        self.d_output = model_setup['d_output']
        self.output_seq = model_setup['output_seq']
        self.super_init = False


    def forward(self, inpt):
        #inpt N,S,D
        if not self.super_init:
            if not self.output_seq or True:
                self.decoder = nn.Linear(inpt.size(-1), self.d_output)
                self.decoder.to(inpt.device)
            self.super_init = True

        output = self.decoder(inpt)
        if self.output_seq:
            pass
            #output = inpt[:,:,:self.d_output]
        else:
            output = self.decoder(inpt)
            output = output.reshape(-1)

        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)