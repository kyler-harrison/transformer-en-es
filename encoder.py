import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, num_heads):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.QKV_size = input_size  # output size of a Q, K, V are same as input size

        # network layers
        self.Q_layer = nn.Linear(input_size, num_heads * self.QKV_size)  
        self.K_layer = nn.Linear(input_size, num_heads * self.QKV_size)  
        self.V_layer = nn.Linear(input_size, num_heads * self.QKV_size)  
        self.mha_linear = nn.Linear(num_heads * self.QKV_size, input_size)
        self.layer_norm0 = nn.LayerNorm(input_size)
        self.ff_linear0 = nn.Linear(input_size, input_size)
        self.ff_relu = nn.ReLU(input_size)
        self.ff_linear1 = nn.Linear(input_size, input_size)
        self.layer_norm1 = nn.LayerNorm(input_size)


    def forward(self, inputs, p_dropout=0.1):
        # p>0 for training, p=0 for inference
        # this layer could be init w the nn if i did this in the nice torch way
        dropout = nn.Dropout(p=p_dropout)

        # inputs are embedding + positional encoding
        inputs = dropout(inputs)

        # multi-headed attention start
        # Q, K, V linear layer outputs (tensors)
        Q = self.Q_layer(inputs)
        K = self.K_layer(inputs)
        V = self.V_layer(inputs)

        # scaled dot-product attention
        sdpa = torch.bmm(torch.softmax(torch.bmm(Q, torch.transpose(K, 1, 2)) / np.sqrt(self.QKV_size), dim=2), V)

        mha_out = self.mha_linear(sdpa)
        mha_out = dropout(mha_out)
        # multi-headed attention end

        # add & norm (norm operates across row in batch tensor's sub-matrices)
        mha_out_anorm = self.layer_norm0(mha_out + inputs)

        # feed fwd
        ff_output = self.ff_linear0(mha_out_anorm)
        ff_output = self.ff_relu(ff_output)
        ff_output = self.ff_linear1(ff_output)
        ff_output = dropout(ff_output)

        # add & norm
        encoder_output = self.layer_norm1(ff_output + mha_out_anorm)

        return encoder_output
