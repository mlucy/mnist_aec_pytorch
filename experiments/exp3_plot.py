import sys
import pandas as pd
import torch
import math

import matplotlib as mpl
from matplotlib import pyplot as plt

res = torch.load(sys.argv[1])
# RSI: remove
# res = pd.DataFrame([{'eps': 0.3, 'norm': 'linf', 'random_start': False, 'name': '0', 'adversary': 'FGSM', 'accuracy': 0.9599999785423279}, {'eps': 0.3, 'norm': 'linf', 'k': 40, 'a': 0.01, 'random_start': True, 'restarts': 1, 'name': '1', 'adversary': 'PGD', 'accuracy': 0.9549999833106995}, {'eps': 0.3, 'norm': 'linf', 'k': 100, 'a': 0.01, 'random_start': True, 'restarts': 1, 'name': '2', 'adversary': 'PGD', 'accuracy': 0.949999988079071}, {'eps': 0.3, 'norm': 'linf', 'k': 40, 'a': 0.01, 'random_start': True, 'restarts': 40, 'name': '3', 'adversary': 'PGD', 'accuracy': 0.9549999833106995}, {'eps': 0.3, 'norm': 'linf', 'k': 100, 'a': 0.01, 'random_start': True, 'restarts': 40, 'name': '4', 'adversary': 'PGD', 'accuracy': 0.9549999833106995}, {'eps': 0.3, 'norm': 'linf', 'random_start': False, 'model': "A'", 'name': '5', 'adversary': 'FGSM', 'accuracy': 0.9699999690055847}, {'eps': 0.3, 'norm': 'linf', 'k': 40, 'a': 0.01, 'random_start': True, 'model': "A'", 'restarts': 1, 'name': '6', 'adversary': 'PGD', 'accuracy': 0.9649999737739563}, {'eps': 0.3, 'norm': 'linf', 'k': 100, 'a': 0.01, 'random_start': True, 'model': "A'", 'restarts': 20, 'name': '7', 'adversary': 'PGD', 'accuracy': 0.9649999737739563}])
print(res)

title_opts = {'y': -0.3, 'fontdict': {'fontsize': 16}}

fig = plt.figure()
ax = fig.gca()
ax.axis('off')
ax.axis('tight')

res = res[['adversary', 'k', 'restarts', 'model', 'accuracy', 'theirs']]
res.update(res[['model']].applymap(
    lambda x: 'A' if isinstance(x, float) and math.isnan(x) else x
))
res.update(res.applymap(lambda x: '-' if isinstance(x, float) and math.isnan(x) else x))
res.update(res[['accuracy']].applymap(lambda x: f'{x:.3f}'))
res.update(res.applymap(
    lambda x: f'{x:g}' if isinstance(x, float) else x
))
print(res)
ax.table(res.values, colLabels=res.columns, loc='center')
fig.tight_layout()
fig.savefig('results/exp3.png')

