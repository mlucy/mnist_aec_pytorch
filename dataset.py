import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from torch.utils.data import DataLoader, get_worker_info, TensorDataset
import functools
import os

from torch import multiprocessing
multiprocessing.set_start_method('fork')

import utils
log = utils.get_logger(__name__)

def mnist_iterator(mnist, train=True):
    dataset = mnist['train' if train else 'test']
    to_tensor = transforms.ToTensor()
    for i in range(len(dataset)):
        worker_info = get_worker_info()
        if worker_info is not None:
            if worker_info.id != i % worker_info.num_workers:
                continue
        yield (
            to_tensor(dataset[i]['image']),
            torch.tensor(dataset[i]['label']),
        )

def tensor_preload(n, iterator):
    xs = None
    ys = None
    i = 0
    for x, y in iterator:
        if xs is None:
            xs = torch.empty(n, *x.shape, dtype=x.dtype)
        if ys is None:
            ys = torch.empty(n, *y.shape, dtype=y.dtype)
        xs[i] = x
        ys[i] = y
        i += 1
    assert i == n
    return xs.to('cuda'), ys.to('cuda')

CACHED_MNIST_TENSORS = None
def get_mnist_tensors(train=True):
    global CACHED_MNIST_TENSORS
    cache_path = f'/tmp/_mnist_aec_cache.pt'
    if CACHED_MNIST_TENSORS is None:
        log.info('Loading MNIST tensors...')
        if os.path.exists(cache_path):
            CACHED_MNIST_TENSORS = torch.load(cache_path)
        else:
            log.info('No cached MNIST tensors, constructing...')
            mnist = load_dataset('mnist')
            mnist = mnist.shuffle()
            CACHED_MNIST_TENSORS = [
                tensor_preload(
                    len(mnist['train' if b else 'test']),
                    mnist_iterator(mnist, train=b),
                )
                for b in [False, True]
            ]
            torch.save(CACHED_MNIST_TENSORS, cache_path)
    return CACHED_MNIST_TENSORS[train]

# We're doing so little computation for each batch that we're
# bottlenecked on moving stuff to the GPU, so it's faster to just
# precompute the whole epoch, move it to the GPU, and then use a
# shuffle buffer.
def mnist_dataloader(batch_sz, train=True, shuffle=True):
    ds = TensorDataset(*get_mnist_tensors(train=train))
    if shuffle:
        ds = ShufflerIterDataPipe(ds)
    dl = DataLoader(ds, batch_size=batch_sz)
    return dl
