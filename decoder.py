import torch
import torch.nn as nn
import numpy as np


class Decoder(nn.Module):
    def __init__(self, decoder_input_size, num_heads):
        super(Decoder, self).__init__()
        self.decoder_input_size = decoder_input_size
        self.num_heads = num_heads
        self.QKV_size = decoder_input_size  

        # masked multi-head attention sublayer layers
        self.Q_layer = nn.Linear(decoder_input_size, num_heads * self.QKV_size)  
        self.K_layer = nn.Linear(decoder_input_size, num_heads * self.QKV_size)  
        self.V_layer = nn.Linear(decoder_input_size, num_heads * self.QKV_size)  
        self.mmha_linear = nn.Linear(num_heads * self.QKV_size, decoder_input_size)
        self.layer_norm0 = nn.LayerNorm(decoder_input_size)

        # multi-head attention sublayer layers (2nd attention layer)
        self.Q2_layer = nn.Linear(decoder_input_size, num_heads * self.QKV_size)  
        self.K2_layer = nn.Linear(decoder_input_size, num_heads * self.QKV_size)  
        self.V2_layer = nn.Linear(decoder_input_size, num_heads * self.QKV_size)  
        self.mha_linear = nn.Linear(num_heads * self.QKV_size, decoder_input_size)
        self.layer_norm1 = nn.LayerNorm(decoder_input_size)

        # feed fwd sublayer layers
        self.ff_linear0 = nn.Linear(decoder_input_size, decoder_input_size)
        self.ff_relu = nn.ReLU(decoder_input_size)
        self.ff_linear1 = nn.Linear(decoder_input_size, decoder_input_size)
        self.layer_norm2 = nn.LayerNorm(decoder_input_size)


    def forward(self, decoder_inputs, encoder_context, og_inputs, mask=True):
        # masked multi-head attention start
        # Q, K, V linear layer decoder_inputs
        Q = self.Q_layer(decoder_inputs)
        K = self.K_layer(decoder_inputs)
        V = self.V_layer(decoder_inputs)

        # scaled dot-product attention
        sdpa0 = torch.bmm(Q, torch.transpose(K, 1, 2)) / np.sqrt(self.QKV_size)
        if mask:
            mask = -1e6 * torch.triu(torch.ones_like(sdpa0), diagonal=1)
            sdpa0 = torch.tril(sdpa0) + mask
        sdpa0 = torch.softmax(sdpa0, dim=2)
        sdpa0 = torch.matmul(sdpa0, V)

        mmha_out = self.mmha_linear(sdpa0)
        # masked multi-head attention end

        # add & norm 0 
        mmha_anorm = self.layer_norm0(mmha_out + decoder_inputs)

        # multi-head attention start
        # Q input: decoder's mmha_anorm, K: encoder_context, V: encoder_context 
        Q2 = self.Q2_layer(mmha_anorm)
        K2 = self.K2_layer(encoder_context)
        V2 = self.V2_layer(encoder_context)
        sdpa1 = torch.bmm(torch.softmax(torch.bmm(Q2, torch.transpose(K2, 1, 2)) / np.sqrt(self.QKV_size), dim=2), V2)
        mha_out = self.mha_linear(sdpa1)
        # multi-head attention end

        # add & norm 1
        mha_anorm = self.layer_norm1(mha_out + mmha_anorm)

        # feed fwd network
        ff_output = self.ff_linear0(mha_anorm)
        ff_output = self.ff_relu(ff_output)
        ff_output = self.ff_linear1(ff_output)

        # add & norm 2
        decoder_output = self.layer_norm2(ff_output + mha_anorm)

        return decoder_output
