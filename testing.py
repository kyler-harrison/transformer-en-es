#!/bin/python3

import torch
import torch.nn.functional as F
import random
from transformer import Transformer


def save_data(enc_inp, dec_inp, targets, enc_path, dec_path, target_path):
    torch.save(enc_inp, enc_path)
    torch.save(dec_inp, dec_path)
    torch.save(targets, target_path)


def load_data(enc_path, dec_path, target_path):
    enc_inp = torch.load(enc_path)
    dec_inp = torch.load(dec_path)
    targets = torch.load(target_path)

    return enc_inp, dec_inp, target_path


def test(tfrmr, enc_inp, dec_inp, targets):
    outs = tfrmr.forward(enc_inp, dec_inp)
    print(targets)
    print(outs)
    loss = tfrmr.calc_loss(outs, targets)
    print(loss)
    tfrmr.backward()


def main():
    # params to change
    num_tokens = 7
    embedding_len = 5
    batch_size = 3
    target_vocab_size = 10
    choices = [0.0, 1.0]
    create_rando = True
    save = True

    # load/create dummy data
    if create_rando:
        enc_inp = torch.tensor([[[random.choice(choices) for j in range(embedding_len)] for i in range(num_tokens)] for b in range(batch_size)])
        dec_inp = torch.tensor([[[random.choice(choices) for j in range(embedding_len)] for i in range(num_tokens)] for b in range(batch_size)])

        targets = torch.zeros(batch_size, num_tokens, target_vocab_size)
        for b in range(batch_size):
            target = F.one_hot(torch.arange(0, num_tokens), num_classes=target_vocab_size).double()
            targets[b][:num_tokens] = target

        if save:
            save_data(enc_inp, dec_inp, targets, "enc_inp0.pt", "dec_inp0.pt", "targets0.pt")
    else:
        enc_path = "enc_inp0.pt"
        dec_path = "dec_inp0.pt"
        target_path = "targets0.pt"
        enc_inp, dec_inp, targets = load_data(enc_path, dec_path, target_path)

    # init transformer
    N_layers = 1
    n_heads = 1
    tfrmr = Transformer(embedding_len, target_vocab_size, N_layers, n_heads)

    # tests
    test(tfrmr, enc_inp, dec_inp, targets)


if __name__ == "__main__":
    main()
