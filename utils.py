import os
from pathlib import Path
import numpy as np
import torch

from glove import GloveModel
from datasets import GloveDataset

def __load_ds(ds_path, n_words: int=10000000, window_size: int= 5):
    with open(ds_path) as f:
        dataset = GloveDataset(f.read(), n_words, window_size)
    return dataset

def __load_model(weights, vocab_size, embed_dim: int=300):
    device = torch.device('cpu')
    model = GloveModel(vocab_size, embed_dim)
    model = model.eval().to(device)
    
    model.load_state_dict(torch.load(weights, map_location=device))
    return model
    
def setup(ds_path, weights, n_words: int=10000000, window_size: int=5, embed_dim: int=300):
    dataset = __load_ds(ds_path, n_words, window_size)
    model = __load_model(weights, dataset._vocab_len, embed_dim)
    
    return dataset, model

def similar_word(word: str, model: GloveModel, dataset: GloveDataset, top: int=10):
    
    if word not in dataset._tokens:
        print('No such a word in the vocabulary')
        return None
    
    word_id = dataset._word2id[word]
    similar_words = model.similar_words_by_id(word_id)
    
    res = [(dataset._id2word[w[0]], w[1]) for w in similar_words]
    return res[1:top+1]
