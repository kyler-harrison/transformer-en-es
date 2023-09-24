import torch


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
    pos_matrix = torch.zeros(num_words, embedding_len)
    d_model = num_tokens * embeddingg_len

    for pos in range(num_tokens):
        for i in range(embedding_len):
            if (i % 2 == 0):
                pos_matrix[pos, i] = np.sin(pos / 10000**(2 * i / d_model))
            else:
                pos_matrix[pos, i] = np.cos(pos / 10000**(2 * i / d_model))

    return pos_matrix
