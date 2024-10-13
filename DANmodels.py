from torch.utils.data import Dataset
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import Counter, defaultdict


#Custom dataset to hold data for Dan models
class DanData(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        words, label = self.examples[idx].words, self.examples[idx].label
        return words, label

class DAN(nn.Module):
    def __init__(self, embedding_layer, embedding_dim, hidden_dim):
        super(DAN, self).__init__()
        self.embedding = embedding_layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2) 
        self.relu = nn.ReLU()

    def forward(self, x):
        embeddings = self.embedding(x)  
        avg_embedding = torch.mean(embeddings, dim=1)  # Averaging
        hidden_output = self.relu(self.fc1(avg_embedding))
        logits = self.fc2(hidden_output)
        return logits    

class DANSUB(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(DANSUB, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2) 
        self.relu = nn.ReLU()

    def forward(self, x):
        embeddings = self.embedding(x)  
        avg_embedding = torch.mean(embeddings, dim=1)  # Averaging
        hidden_output = self.relu(self.fc1(avg_embedding))
        logits = self.fc2(hidden_output)
        return logits    
# with dropout
# class DAN(nn.Module):
#     def __init__(self, embedding_layer, embedding_dim, hidden_dim, dropout_prob = 0.5):
#         super(DAN, self).__init__()
#         self.embedding = embedding_layer
#         self.fc1 = nn.Linear(embedding_dim, hidden_dim)
#         self.dropout = nn.Dropout(dropout_prob)  # Dropout layer
#         self.fc2 = nn.Linear(hidden_dim, 2)  # Binary classification
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         embeddings = self.embedding(x)  # shape: (batch_size, seq_len, embedding_dim)
#         avg_embedding = torch.mean(embeddings, dim=1)  # Average along sequence length
#         hidden_output = self.relu(self.fc1(avg_embedding))
#         hidden_output = self.dropout(hidden_output)  # Apply dropout
#         logits = self.fc2(hidden_output)
#         return logitsz

#heloping function to convert words to indices
def collate_fn(batch, word_indexer):
    # Convert words to indices and pad them
    max_len = max(len(words) for words, _ in batch)
    word_indices = []
    labels = []
    for words, label in batch:
        indices = [word_indexer.index_of(word) if word_indexer.index_of(word) != -1 else word_indexer.index_of("UNK") for word in words]
        indices += [word_indexer.index_of("PAD")] * (max_len - len(indices))
        word_indices.append(indices)
        labels.append(label)

    return torch.tensor(word_indices, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# BPE implementation
class BPE:
    def __init__(self, vocab, num_merges):
        self.vocab = vocab
        self.num_merges = num_merges
        self.bpe_codes = {}
        self.word2index = {}

    def get_stats(self):
        pairs = Counter()
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair):
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        new_vocab = {}
        for word in self.vocab:
            new_word = pattern.sub(''.join(pair), word)
            new_vocab[new_word] = self.vocab[word]
        return new_vocab

    def learn_bpe(self):
        for i in range(self.num_merges):
            pairs = self.get_stats()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.vocab = self.merge_vocab(best_pair)
            self.bpe_codes[best_pair] = i
        self.build_word2index()
        return self.bpe_codes

    def tokenize(self, word):
        if ' ' not in word:
            word = ' '.join(list(word))
        while True:
            pairs = self.get_stats()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            new_word = word.replace(' '.join(best_pair), ''.join(best_pair))
            if new_word == word:
                break
            word = new_word
        return word.split()
#mapping the word to index
    def build_word2index(self):
        index = 0
        for word in self.vocab:
            tokens = word.split()
            for token in tokens:
                if token not in self.word2index:
                    self.word2index[token] = index
                    index += 1

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.word2index[token] for token in tokens if token in self.word2index]
    # optional decode method
    def decode(self, indices):
        tokens = [self.index2word[index] for index in indices if index in self.index2word]
        return ' '.join(tokens)