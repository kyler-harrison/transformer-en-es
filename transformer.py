import torch
import torch.nn as nn
import torch.optim as optim
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, size, final_out_size, N_layers, num_heads, optimizer_lr=0.0001):
        """
        Following the architecture given by: 
        https://arxiv.org/pdf/1706.03762.pdf
        """
        super(Transformer, self).__init__()

        # NOTE: paper mentions d_model a lot. i think this is the size of
        # one embdedded sentence. so with a word2vec embedding of 100
        # and 20 max words in a sentence, d_model = 20 * 100 = 2000
        # although im not super sure...

        # input size and output size should be same 
        self.size = size
        self.final_out_size = final_out_size
        self.num_heads = num_heads

        self.encoders = nn.ModuleList([Encoder(size, num_heads) for i in range(N_layers)])
        self.decoders = nn.ModuleList([Decoder(size, num_heads) for i in range(N_layers)])
        self.linear_layer = nn.Linear(size, final_out_size)
        
        self.optimizer = optim.Adam(self.parameters(), lr=optimizer_lr, betas=(0.9, 0.98), eps=1e-9)
        self.loss_criteria = nn.CrossEntropyLoss()
        self.loss = None


    def forward(self, encoder_inputs, decoder_inputs, encoder_context=None, encoder_only=False, decoder_only=False):
        # based on conditionals, pass through all encoders and then all
        # decoders

        # if decoder_only, encoder_context should be passed in
        # encoder_only and decoder_only should only be set to True when 
        # doing inference (both should not be True as input)

        if (encoder_only and decoder_only):
            print("hey what the hell man encoder_only and decoder_only can't both be True")
            return None

        elif (encoder_only):
            # called when doing inference (encoder context needs to be created once)
            encoder_context = encoder_inputs
            for encoder in self.encoders:
                encoder_context = encoder.forward(encoder_context)

            return encoder_context

        elif (decoder_only):
            # called when doing inference (encoder context already created)
            for decoder in self.decoders:
                decoder_inputs = decoder.forward(decoder_inputs, encoder_context, encoder_inputs, mask=True)

            linear_out = self.linear_layer(decoder_inputs)

            return linear_out

        else:
            # called when training (need to pass through all encoders and decoders)
            encoder_context = encoder_inputs
            for encoder in self.encoders:
                encoder_context = encoder.forward(encoder_context)

            decoder_inputs = decoder_inputs
            for decoder in self.decoders:
                decoder_inputs = decoder.forward(decoder_inputs, encoder_context, encoder_inputs)

            linear_out = self.linear_layer(decoder_inputs)
            
            return linear_out


    def calc_loss(self, transformer_outputs, targets, test_loss=False):
        # CrossEntropyLoss() combines softmax with loss calculation.
        # to compute the loss for batches of matrices, must reshape into
        # one big matrix. CrossEntropyLoss() requires this shape.
        # loss is calculated against each predicted word vector, and then 
        # averaged across all of the words predicted in the batch. 
        # to get actual probs, run a softmax on the final output of the
        # transformer's forward pass instead. 
        batch_size = transformer_outputs.shape[0] * transformer_outputs.shape[1]
        num_classes = transformer_outputs.shape[2]
        transformer_outputs_reshaped = transformer_outputs.view(batch_size, num_classes)
        targets_reshaped = targets.view(batch_size, num_classes)

        # compute loss
        loss = self.loss_criteria(transformer_outputs_reshaped, targets_reshaped)

        # dont update model's loss if just checking against test data
        if test_loss:
            return loss
        else:
            self.loss = loss
            return loss


    def backward(self):
        # autograd handles everything i think... 
        # update params
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

