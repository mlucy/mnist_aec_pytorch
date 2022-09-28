import sys
import pandas as pd
import torch

import matplotlib as mpl
from matplotlib import pyplot as plt

res = torch.load(sys.argv[1])
print(res)

scales = [1, 2, 4, 8, 16]
adversaries = ['ID', 'FGSM', 'PGD']
adversary_colors = {
    'ID': 'blue',
    'FGSM': 'red',
    'PGD': 'black',
}
adversary_names = {
    'ID': 'Standard',
    'FGSM': 'FGSM',
    'PGD': 'PGD',
}

title_opts = {'y': -0.3, 'fontdict': {'fontsize': 16}}

def adversary_graph(subp, train_adversary):
    subp.set_ylabel('Accuracy')
    subp.set_xlabel('Capacity Scale')
    subp.set_title(f'{adversary_names[train_adversary]} training', **title_opts)
    adv_res = res[res.train_adversary == train_adversary]
    for adversary in adversaries:
        xs = [str(x) for x in scales]
        ys = [float(adv_res[adv_res.scale == x][f'{adversary}_accuracy']) for x in scales]
        subp.plot(xs, ys, color=adversary_colors[adversary])

fig = plt.figure(figsize=(16, 4))
for i, adversary in enumerate(adversaries):
    ax = fig.add_subplot(1, 4, i+1)
    adversary_graph(ax, adversary)

ax = fig.add_subplot(1, 4, 4)
ax.set_ylabel('Average Loss')
ax.set_yscale('log')
ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
ax.set_xlabel('Capacity Scale')
ax.set_title('Training loss', **title_opts)
for adversary in adversaries:
    xs = [str(x) for x in scales]
    adv_res = res[res.train_adversary == adversary]
    ys = [
        float(adv_res[adv_res.scale == x][f'{adversary}_loss'])
        for x in scales
    ]
    kw = {'label': adversary_names[adversary]}
    ax.plot(xs, ys, color=adversary_colors[adversary], **kw)

fig.legend()
fig.tight_layout()
fig.savefig('results/exp1.png')

