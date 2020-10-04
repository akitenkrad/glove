import os
from pathlib import Path
from time import time
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datasets import GloveDataset
from glove import GloveModel, weight_func, wmse_loss

def run_train(ds_path: str, outdir:str='output', logdir: str='logs'):
    N_WORDS = 10000000
    EMBED_DIM = 300
    N_EPOCHS = 100
    BATCH_SIZE = 2048
    X_MAX = 100
    ALPHA = 0.75
    
    with open(ds_path) as f:
        dataset = GloveDataset(f.read(), N_WORDS)
    glove = GloveModel(dataset._vocab_len, EMBED_DIM)
    optimizer = optim.Adagrad(glove.parameters(), lr=0.05)
    
    st = time()
    n_batches = int(len(dataset._xij) / BATCH_SIZE)
    loss_values = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    glove.train().to(device)
    
    os.makedirs(str(Path(outdir)), exist_ok=True)
    os.makedirs(str(Path(logdir)), exist_ok=True)
    
    history = SummaryWriter(log_dir=logdir)
    
    with tqdm(total=N_EPOCHS) as epoch_iter:
        for epoch in range(0, N_EPOCHS):
            batch_i = 0

            with tqdm(total=n_batches, desc='Training GloVe', leave=None) as batch_iter:
                for x_ij, i_idx, j_idx in dataset.get_batches(BATCH_SIZE):
                    batch_i += 1

                    optimizer.zero_grad()

                    outputs = glove(i_idx, j_idx)
                    weights_x = weight_func(x_ij, X_MAX, ALPHA, device)
                    loss = wmse_loss(weights_x, outputs, torch.log(x_ij), device)

                    loss.backward()
                    optimizer.step()

                    loss_values.append(loss.item())

                    batch_log = 'Epoch: {}/{}  Batch: {}/{}  Loss: {:.3f}  eTime: {:.1f}m'.format(
                            epoch + 1, N_EPOCHS, batch_i, n_batches, np.mean(loss_values[-20:]), (time() - st) / 60.0)
                    batch_iter.update(1)
                    batch_iter.set_description(batch_log)

            epoch_log = 'Loss: {:.3f}  eTime: {:.1f}m'.format(np.mean(loss_values[-20:]), (time() - st) / 60.0)
            epoch_iter.update(1)
            epoch_iter.set_description(epoch_log)

            history.add_scalar('train-loss', np.mean(loss.item()), epoch + 1)

            print('Saving model...')
            torch.save(glove.state_dict(), Path(outdir) / 'glove.pth')
        
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--ds-path', type=str, help='dataset path')
    parser.add_argument('--outdir', type=str, help='output directory')
    parser.add_argument('--logdir', type=str, help='log directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = build_parser()
    
    run_train(args.ds_path, args.logdir)
    