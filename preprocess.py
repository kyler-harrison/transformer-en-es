import torch
import numpy as np


def sentence_to_tokens(sent, language):
    """
    Prepares input sentence and splits it into list of tokens. 
    Puts spaces around punctuation so split captures them as
    tokens, lowercases string, splits by tokens, adds start and
    end token to list.

    sent (str): string of a sentence
    language (str): either "en" or "es"

    return (list): tokenized list of sent
    """

    if (language == "en"):
        punctuations = [',', '.', '!', '?', '"', "-"]
    elif (language == "es"):
        punctuations = [',', '.', '!', '?', '¡', '¿', '"', '-']
    else:
        print("language argument must be 'en' or 'es'")
        return None

    for p in punctuations:
        sent = sent.replace(p, f" {p} ")

    sent = sent.lower()
    tokens = sent.split()
    start_token = "<s>"
    end_token = "<e>"
    tokens = [start_token] + tokens + [end_token]

    return tokens


def vec_str_to_list(vec_df, vec_col):
    """
    word2vec vectors are stored in csv as strings. This function transforms the vectors into
    actual lists of floats.
    """

    vec_series = vec_df[vec_col].apply(lambda vec_str: [float(val) for val in vec_str[1:-1].split()])
    return vec_series


def tokens_to_tensor(tokens, num_tokens, embedding_len, vec_df, vec_token_col, vec_embed_col):
    embedded_tensor = torch.zeros(num_tokens, embedding_len)

    for i, token in enumerate(tokens):
        # previous data prep steps ensure there will be 1 match here (hence taking 0th is safe)
        token_vec = vec_df[vec_df[vec_token_col] == token][vec_embed_col].values[0]
        embedded_tensor[i] = torch.tensor(token_vec)

    return embedded_tensor


def embed_input_batch(batch_df, tokens_col, vec_df, vec_token_col, vec_embed_col, embedding_len, num_tokens):
    """
    Returns a tensor containing embedded vector representations
    of the sequences in batch_df.

    input:
    batch_df: dataframe containing a column (tokens_col) with a list of tokens
              representing a sequence (assumed to be sentences)
    tokens_col: name of column containing tokenized sequence lists to operate on 
    vec_df: dataframe containing word embedding vectors
    vec_token_col: name of column containing string tokens in vec_df
    vec_embed_col: name of vector column containing word2vec embeddings in vec_df
    embedding_len: length of the vectors in vec_df
    num_tokens: maximum number of tokens in sequence (matrices padded to this number of rows)
    """

    return torch.stack(batch_df[tokens_col].apply(lambda tokens: tokens_to_tensor(tokens, num_tokens, embedding_len, vec_df, vec_token_col, vec_embed_col)).values.tolist())


def gen_positional_encoding(num_tokens, embedding_len):
    """
    Creates positional encoding matrix of shape (num_tokens, embedding_len)

    num_tokens: max number of tokens in input sequence matrix
    embedding_len: length of embedded token vector
    example:
    let sequence matrix be (num_tokens=4, embedding_len=3):
    [[0.03, 0.02, 0.01],  -> pos=0, i=0,1,2
     [0.04, 0.03, 0.02],  -> pos=1, i=0,1,2
     [0.05, 0.04, 0.03],  -> pos=2, i=0,1,2
     [0.06, 0.05, 0.04]]  -> pos=3, i=0,1,2
    positonal encoding matrix (values not actually calculated):
    [[1, 2, 3],
     [5, 6, 7],
     [4, 5, 6],
     [7, 8, 9]
    add em up elem-wise to get positionally-encoded matrix (not done in this fun):
    [[1.03, 2.02, 3.01],
     [5.04, 6.03, 7.02],
      ...]
    returns matrix of positional encoding values to be added to sequence matrices
    """
    pos_matrix = torch.zeros(num_tokens, embedding_len)
    d_model = num_tokens * embedding_len

    for pos in range(num_tokens):
        for i in range(embedding_len):
            if (i % 2 == 0):
                pos_matrix[pos, i] = np.sin(pos / 10000**(2 * i / d_model))
            else:
                pos_matrix[pos, i] = np.cos(pos / 10000**(2 * i / d_model))

    return pos_matrix


def pos_encode_batch(batch_data, sequence_lens, pos_encoding_mat):
    """
    Given batch of data, add positional encoding matrix to each sub-matrix
    only where there is data (i.e. don't add to 0 rows).

    batch_data: tensor batch of data with shape like (num_batches, num_tokens, embedding_len)
    sequence_lens: number of tokens in each sequence in batch (NOTE must be same order as batch)
    pos_encoding_mat: positional encoding matrix generated with other preprocess function

    Operates on batch data inplace. Returns None.
    """

    for i, sequence_len in enumerate(sequence_lens):
        batch_data[i][:sequence_len] += pos_encoding_mat[:sequence_len]


def one_hot_tokens(tokens, vec_df, num_tokens, output_size, vec_token_col, smoothing=False, smoothing_epsilon=0.1):
    """
    Given list of tokens and other args defined in one_hot_batch, creates a matrix of one-hot
    vectors.

    Returns a tensor
    """

    if (smoothing):
        # not sure if these even makes a diff for 0s, since smoothed val is probs order of 1e-6 
        smoothed_val = smoothing_epsilon / (output_size - 1)
        out_tensor = torch.zeros(num_tokens, output_size, smoothed_val)
    else:
        out_tensor = torch.zeros(num_tokens, output_size)

    # NOTE <s> token not included in output target (hence the [1:])
    # <e> is included and nothing special needs to be done here
    for i, token in enumerate(tokens[1:]):
        # find token's index (vec_df is token->index mapping), should be guaranteed to be found
        token_idx = vec_df[vec_df[vec_token_col] == token].index.values[0]

        # set one-hot at token's index
        if (smoothing):
            out_tensor[i][token_idx] = 1.0 - smoothing_epsilon
        else:
            out_tensor[i][token_idx] = 1.0

    return out_tensor


def one_hot_batch(batch_df, vec_df, num_tokens, output_size, tokens_col, vec_token_col, smoothing=False, smoothing_epsilon=0.1):
    """
    Given a dataframe of a batch, returns a tensor batch where each sub-matrix contains rows of
    one-hot token vectors. See https://paperswithcode.com/method/label-smoothing for label
    smoothing explanation.

    batch_df: dataframe containing a column of tokens and a column of their respective lengths
    vec_df: dataframe that maps tokens to vector representations, df index used as one-hot index
    num_tokens: max number of tokens in an output prediction
    output_size: size of single output vector (should be target vocab size, i.e. large)
    tokens_col: name of column of tokens in batch_df
    vec_token_col: name of token column in vec_token_col
    smoothing: bool to smooth labels or not
    smoothing_epsilon: epsilon value to use in label smoothing 

    Returns tensor of matrices containing one-hot vectors.
    """
    return torch.stack(batch_df[tokens_col].apply(lambda tokens: one_hot_tokens(tokens, vec_df, num_tokens, output_size, vec_token_col)).values.tolist())
    

