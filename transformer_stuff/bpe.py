import tiktoken

import re
import collections

class BPE:
    def __init__(self, num_merges):
        self.num_merges = num_merges

    def get_vocab(self, text):
        vocab = collections.defaultdict(int)
        for word in text.split():
            vocab[' '.join(word)] += 1
        return vocab

    def get_stats(self, vocab):
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = pattern.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def extract_tokens(self, vocab):
        tokens = set()
        for word in vocab.keys():
            tokens.update(word.split())
        return tokens

    def encode(self, text):
        vocab = self.get_vocab(text)
        for _ in range(self.num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)

        tokens = self.extract_tokens(vocab)
        return tokens

    def decode(self, tokens):
        return ' '.join(tokens).replace('▁', ' ').strip()

datafile = '../data/shakespeare.txt'
with open(datafile, 'r') as f:
    bpe_encoder = BPE(num_merges=1000)
    text = f.read()
    tokens = bpe_encoder.encode(text)
    print(tokens)
    print(bpe_encoder.decode(tokens))