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

