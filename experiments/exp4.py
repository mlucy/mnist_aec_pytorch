import pandas as pd
import copy
import pprint
import torch
import os
import numpy as np

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import utils
utils.log_setup()
log = utils.get_logger(__name__)

import trainer

os.makedirs('results', exist_ok=True)
respath = 'results/exp4.df'

log.info('Computing results...')

default_config = utils.load_config('default')
results = []
for adversary in [None, 'PGD']:
    config = copy.deepcopy(default_config)
    # config['trainer']['max_train_steps'] = 2000
    if adversary is not None:
        config['trainer']['train_adversary'] = {
            **config['adversaries'][adversary],
            'norm': norm,
        }
    norm_ranges = {
        'linf': np.arange(0, 0.5, 0.025),
        'l2': np.arange(0, 6.5, 0.5),
    }
    config['eval']['final_metrics'] = []
    for norm in ['l2', 'linf']:
        for eps in norm_ranges[norm]:
            config['eval']['final_metrics'].append({
                'adversaries': [{
                    **config['adversaries']['PGD'],
                    'norm': norm,
                    'eps': eps,
                    'k': 100,
                    'a': 2.5 * eps / 100,
                }],
                'metrics': ['accuracy'],
                'tags': {'norm': norm, 'eps': eps},
            })
    del config['adversaries']
    log.info('-'*20)
    log.info('NEW RUN')
    log.info('-'*20)
    log.info('\n'+pprint.pformat(config))
    log.info('---')
    results.extend(trainer.eval(config))
log.info(results)
df = pd.DataFrame(results)
log.info(df)

res = df
torch.save(res, respath)
log.info(f'Saved to {respath}.')



