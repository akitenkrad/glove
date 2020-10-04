# PyTorch implementation of GloVe

this repository is implemented based on https://nlpython.com/implementing-glove-model-with-pytorch

## download dataset

- [text8 english](https://drive.google.com/file/d/15ygx-KkU9heoFO-o5R_odp_4xKR31wrB/view?usp=sharing)


## train
```bash
> python -m train --ds-path datasets/text8 --outdir outputs --logdir logs
```

## show logs
```bash
> tensorboard --logdir logs
```
