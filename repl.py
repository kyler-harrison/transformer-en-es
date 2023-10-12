#!/bin/python3 

import torch
import pandas as pd
import preprocess as prp
from transformer import Transformer


def load_model(model, model_name):
    """
    Load saved pytorch model and metrics created with save_model().
    Must initialize model with same params as one you load from path.
    
    returns torch model, list of train losses, list of test losses, list of bleu scores
    """
    
    # create paths
    model_path = f"models/{model_name}.pt"
    
    # load model
    model.load_state_dict(torch.load(model_path))
        
    return model


def check_tokens(tokens, vec_df):
    """
    Returns a list containing invalid tokens found in tokens input list.
    """

    tokens_df = pd.DataFrame({"word": tokens})
    tokens_df["valid"] = tokens_df["word"].isin(vec_df["word"])
    return tokens_df[tokens_df["valid"] == False]["word"].values.tolist()


def inference(tfrmr, test_batch, en_vec_df, es_vec_df, batch_size, num_tokens, embedding_len,
              output_size, pos_encoding_mat, start_token_vec, end_token_idx, device):
    """
    This function generates a predicted sequence for each input sequence in input batch iteratively.
    Used for making predictions after training (i.e. when the full target sequence is not available to
    input to the decoder).

    returns a batch of predicted tfrmr outputs (not softmax probs) and a batch of one-hot target sequences
    """

    # transform strings into tensors of embedded values
    encoder_input = prp.embed_input_batch(test_batch, "en_tokens", en_vec_df, "word", "vector", embedding_len, num_tokens).to(device)

    # positionally-encode
    prp.pos_encode_batch(encoder_input, test_batch["en_num_tokens"].values.tolist(), pos_encoding_mat)

    # context generated for every matrix in batch, dont need to do again
    encoder_context = tfrmr.forward(encoder_input, None, encoder_only=True)

    # initialize decoder input for autoregressive prediction
    decoder_input = torch.zeros(batch_size, num_tokens, embedding_len).to(device)

    # holds final sequence prediction for each input sequence
    predicted_seqs = torch.zeros(batch_size, num_tokens, output_size).to(device)

    # iteratively generate predicted sequences for each input sequence
    batch_idx = 0  # this is a hack (artifact from inference loop in train notebook)
    done = False
    token_idx = 0  # starting index of decoder input (0th predicted sequence will be put in decoder_input at token_idx=1 on next iteration)
    decoder_input[batch_idx][token_idx] = start_token_vec  # initialize input with start token

    # WHILE LOOP START
    while ((not done) and (token_idx < (num_tokens - 1))):
        # positional encoding performed row by row
        decoder_input[batch_idx][token_idx] += pos_encoding_mat[token_idx]

        # encoder_context is constant, decoder_input contains data up to before current token
        # and is updated one row at a time with each prediction

        # long statement. makes batches of single matrices so shapes work in transformer
        prediction = tfrmr.forward(encoder_input[batch_idx].view(1, encoder_input.shape[1],
                                      encoder_input.shape[2]),
                                      decoder_input[batch_idx].view(1, decoder_input.shape[1],
                                      decoder_input.shape[2]),
                                      encoder_context=encoder_context[batch_idx].view(1, encoder_context.shape[1],
                                      encoder_context.shape[2]), decoder_only=True)

        # NOTE transformer output is not a softmax distribution across each row
        # softmax is computed w loss calculation. but, the argmax before softmax
        # will be the same as the argmax after softmax (just think about softmax eqn.)
        max_token_idx = prediction[0][token_idx].argmax().item()

        # put prediction into final predicted sequence
        predicted_seqs[batch_idx][token_idx] = prediction[0][token_idx]

        # stop predicting if the end token has been predicted
        if (max_token_idx == end_token_idx):
            done = True
        else:
            # for decoder's input: start token remains at index 0, this prediction goes to token_idx+1
            # for predicted sequence: this prediction goes at token_idx (start token not included in final prediction)
            token_vec = torch.tensor(es_vec_df.loc[max_token_idx, "vector"]).to(device)
            decoder_input[batch_idx][token_idx + 1] = token_vec

        token_idx += 1

        # WHILE LOOP END

    return predicted_seqs


def map_indices_to_tokens(pred_seqs, vec_df):
    """
    Given list of lists containing indices of predicted tokens (output of get_predicted_token_indices()),
    return list of lists where each sublist is now the actual string token corresponding to the input
    index.
    """

    all_preds = []

    for pred_seq in pred_seqs:
        pred_tokens = []

        for pred_token in pred_seq:
            word = vec_df.loc[pred_token]["word"]
            pred_tokens.append(word)

            if (word == "<e>"):
                break

        all_preds.append(pred_tokens)

    return all_preds


def get_predicted_token_indices(predicted_batch):
    """
    Returns a list of lists where each sublist contains the predicted token indices for each
    predicted sequence in predicted_batch.
    """

    pred_seqs = []

    for batch in predicted_batch:
        pred_seq = []

        for row in batch:
            pred_seq.append(row.argmax().item())

        pred_seqs.append(pred_seq)

    return pred_seqs


def get_pred_sent(predicted_tokens):
    """
    Formats list of list predicted_tokens into a string. 
    """

    predicted_str = ""
    pre_punctuations = ['¡', '¿']
    post_punctuations = [',', '.', '!', '?', '"', '-']

    for i, pred in enumerate(predicted_tokens[0][:-1]):
        if (pred in post_punctuations):
            predicted_str += pred
        else:
            if (i == 0):
                predicted_str += f"{pred}"
            else:
                predicted_str += f" {pred}"

    if (predicted_str[0] in pre_punctuations):
        predicted_str = predicted_str[0] + predicted_str[2:]

    return predicted_str


def main():
    # use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load english and spanish dictionaries
    en_vec_df = pd.read_csv("data/cc.en.300.reduced.csv")
    en_vec_df["vector"] = prp.vec_str_to_list(en_vec_df, "vector")
    es_vec_df = pd.read_csv("data/cc.es.300.reduced.csv")
    es_vec_df["vector"] = prp.vec_str_to_list(es_vec_df, "vector")

    # load model
    N_layers = 1
    num_heads = 1
    batch_size = 256
    version = "v3"
    model_name = f"tfrmr_{version}_N={N_layers}_h={num_heads}_B={batch_size}"
    input_size = 300
    output_size = es_vec_df.shape[0]
    max_tokens = 13  # NOTE hardcoded
    tfrmr = Transformer(input_size, output_size, N_layers, num_heads)
    tfrmr = load_model(tfrmr, model_name)
    tfrmr.to(device)

    # infinite REPL
    while True:
        # take input
        print("Input an english sentence:")
        inp = input()

        # tokenize input
        tokenized_inp = prp.sentence_to_tokens(inp, "es")

        # check tokenized length
        if (len(tokenized_inp) > max_tokens):
            print(f"Inputted sequence has length {len(tokenized_inp)} while maximum length is {max_tokens}.\n")
        else:
            # check that all tokens are in dictionary
            invalid_tokens = check_tokens(tokenized_inp, en_vec_df)

            if (len(invalid_tokens) > 0):
                print(f"invalid tokens in input: {invalid_tokens}\n")
            else:
                # prepare input for inference function 
                tokens_df = pd.DataFrame({"en_tokens": [tokenized_inp], "en_num_tokens": [len(tokenized_inp)]})
                pos_encoding_mat = prp.gen_positional_encoding(max_tokens, input_size).to(device)
                start_token_vec = torch.tensor(es_vec_df[es_vec_df["word"] == "<s>"]["vector"].item())
                end_token_idx = es_vec_df[es_vec_df["word"] == "<e>"].index.item()

                # pass input through model
                predicted_seqs = inference(tfrmr, tokens_df, en_vec_df, es_vec_df, 1, 
                                                          max_tokens, input_size, output_size, pos_encoding_mat,
                                                          start_token_vec, end_token_idx, device)

                # transform model output into tokens
                predicted_tokens = map_indices_to_tokens(get_predicted_token_indices(predicted_seqs), es_vec_df)
                predicted_str = get_pred_sent(predicted_tokens)
                print(f"Predicted spanish:\n{predicted_str}\n")


if __name__ == "__main__":
	main()
