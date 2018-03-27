from scipy import sparse
from collections import Counter

def cooccurrence_creator(vocab, corpus, tokensList):
    window_size = 10
    min_count = None

    vocab = Counter()
    for line in tokensList:
        vocab.update(line)
    Dic = {word: (i, freq) for i, (word, freq) in enumerate(vocab.items())}
    id2word = dict((i, word) for word, (i, _) in Dic.items())

    cooccurrence = sparse