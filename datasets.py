import os
from pathlib import Path
from tqdm import tqdm
from collections import Counter, defaultdict
import numpy as np
import torch

class GloveDataset(object):
    def __init__(self, text, n_words=-1, window_size=5):
        self._window_size = window_size
        if n_words < 0:
            self._tokens = [token for token in text.replace('\n', ' ').split(' ')]
        else:
            self._tokens = [token for token in text.replace('\n', ' ').split(' ')[:n_words]]
        word_counter = Counter()
        word_counter.update(self._tokens)
        self._word2id = {w:i for i, (w,_) in enumerate(word_counter.most_common())}
        self._id2word = {i:w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)
        
        self._id_tokens = [self._word2id[w] for w in self._tokens]
        
        self._create_coocurrence_matrix()
        self.batch_size = 1
        
        print('# of words: {}'.format(len(self._tokens)))
        print('Vocabulary length: {}'.format(self._vocab_len))
        
    def _create_coocurrence_matrix(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cooc_mat = defaultdict(Counter)
        for i, w in tqdm(enumerate(self._id_tokens), desc='Create Vocabulary', total=len(self._id_tokens)):
            start_i = max(i - self._window_size, 0)
            end_i = min(i + self._window_size + 1, len(self._id_tokens))
            for j in range(start_i, end_i):
                if i != j:
                    c = self._id_tokens[j]
                    cooc_mat[w][c] += 1 / abs(j - i)
        self._i_idx = []
        self._j_idx = []
        self._xij = []
        
        for w, cnt in tqdm(cooc_mat.items(), desc='Create Coocurrence Matrix', total=len(cooc_mat)):
            for c, v in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(v)
                
        self._i_idx = torch.LongTensor(self._i_idx).to(device)
        self._j_idx = torch.LongTensor(self._j_idx).to(device)
        self._xij = torch.FloatTensor(self._xij).to(device)
        
    def get_batches(self, batch_size):
        self.batch_size = batch_size
        rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))
        
        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]
            