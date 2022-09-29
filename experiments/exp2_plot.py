import sys
import pandas as pd
import torch

import matplotlib as mpl
from matplotlib import pyplot as plt

res = torch.load(sys.argv[1])
print(res)

title_opts = {'y': -0.3, 'fontdict': {'fontsize': 16}}

fig = plt.figure(figsize=(4, 4))
ax = fig.gca()
ax.set_ylabel('Loss Value')
ax.set_yscale('log')
ax.set_xlabel('Iterations')
ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
ax.set_ylim(bottom=0.09, top=1.5)

xs = res['test']['step'].to_numpy()
ys = res['test']['loss'].to_numpy()
ax.plot(xs, ys, color='blue', label='test')

# xs = res['train']['step'].to_numpy()
# ys = res['train']['test_loss'].to_numpy()
# ax.plot(xs, ys, color='red', label='train')

fig.legend()
fig.tight_layout()
fig.savefig('results/exp2.png')

