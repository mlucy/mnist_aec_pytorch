import sys
import pandas as pd
import torch
import math

import matplotlib as mpl
from matplotlib import pyplot as plt

res = torch.load(sys.argv[1])
print(res)

title_opts = {'y': -0.3, 'fontdict': {'fontsize': 16}}

adversary_colors = {
    'PGD': 'blue',
    'ID': 'brown',
}

def plot(ax, norm, data, labels=False):
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('eps')
    ax.set_title(f'{norm} norm', **title_opts)
    for adv in ['ID', 'PGD']:
        subdata = data[data.train_adversary == adv]
        xs = subdata['eps']
        ys = subdata['PGD_accuracy']
        kw = {}
        if labels:
            kw['label'] = 'PGD '
            kw['label'] += ('standard' if adv == 'ID' else 'adv. trained')
        ax.plot(xs, ys, color=adversary_colors[adv], **kw)

fig = plt.figure(figsize=(8, 4))

for i, norm in enumerate(['linf', 'l2']):
    ax = fig.add_subplot(1, 2, i+1)
    plot(ax, norm, res[res.norm == norm], labels=(norm == 'l2'))

fig.legend()
fig.tight_layout()
fig.savefig('results/exp4.png')

