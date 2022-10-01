import sys
import pandas as pd
import torch

import matplotlib as mpl
from matplotlib import pyplot as plt

res = torch.load(sys.argv[1])
print(res)

title_opts = {'y': -0.3, 'fontdict': {'fontsize': 16}}

COLORS = {
    'adam': 'red',
    'paper': 'blue',
}

def stepsize_subgraph(ax, steps, label=False):
    ax.set_ylabel('Accuracy')
    ax.set_ylim(bottom=0.9, top=1.0)
    ax.set_xlabel('Step Size')
    ax.set_xscale('log')
#    ax.set_title(f'{steps} steps', **title_opts)
    # subres = res[res.steps == steps][res.restarts == restarts]
    subres = res[res.steps == steps]
    for step_mode in ['paper', 'adam']:
        stepres = subres[subres.step_mode == step_mode]
        xs = stepres['step_size']
        ys = stepres['accuracy']
        kw = {
            'color': COLORS[step_mode],
        }
        if label:
            kw['label'] = step_mode
        ax.plot(xs, ys, **kw)
        if label:
            ax.legend()

def dists_subgraph(ax, steps, clip, label=False):
    NAMES = {
        'shell_d': 'min dist',
        'shell_du': 'min dist',
        'shell_pd': 'avg dist',
        'shell_pdu': 'avg dist',
    }
    LINESTYLES = {
        'shell_d': 'dashed',
        'shell_du': 'dashed',
        'shell_pd': 'solid',
        'shell_pdu': 'solid',
    }
    if clip:
        ax.set_ylabel('Clipped Linf Distance')
        ax.set_ylim(bottom=0, top=0.1)
    else:
        ax.set_ylabel('Unclipped Linf Distance')
        ax.set_ylim(bottom=0, top=0.3)
    ax.set_xlabel('Step Size')
    ax.set_xscale('log')
    if clip:
        ax.set_title(f'{steps} steps', **title_opts)
    # subres = res[res.steps == steps][res.restarts == restarts]
    subres = res[res.steps == steps]
    if clip:
#        targets = ['shell_d', 'shell_pd']
        targets = ['shell_pd']
    else:
#        targets = ['shell_du', 'shell_pdu']
        targets = ['shell_pdu']
    for target in targets:
        for step_mode in ['paper', 'adam']:
            stepres = subres[subres.step_mode == step_mode]
            xs = stepres['step_size']
            ys = stepres[target]
            kw = {
                'color': COLORS[step_mode],
                'linestyle': LINESTYLES[target],
            }
            if label:
                kw['label'] = f'{step_mode} {NAMES[target]}'
            ax.plot(xs, ys, **kw)
            if label:
                ax.legend()


for adversary in ['PGD', 'ADAM_PGD']:
    res = torch.load(sys.argv[1])
    res = res[res.train_adversary == adversary]
    fig = plt.figure(figsize=(8, 12))
    i = 0

    for steps in [40, 100]:
        ax = fig.add_subplot(3, 2, i+1)
        stepsize_subgraph(ax, steps, label=(i==1))
        i += 1

    for clip in [False, True]:
        for steps in [40, 100]:
            ax = fig.add_subplot(3, 2, i+1)
            dists_subgraph(ax, steps, clip=clip, label=(i in [3, 5]))
            i += 1

    fig.tight_layout()
    fig.savefig(f'results/exp5_{adversary}.png')

