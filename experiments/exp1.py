import pandas as pd
import copy
import pprint
import torch
import os

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import utils
utils.log_setup()
log = utils.get_logger(__name__)

import trainer

os.makedirs('results', exist_ok=True)
respath = 'results/exp1.df'

default_config = utils.load_config('default')
acc_df = pd.DataFrame()
for scale in [1, 2, 4, 8, 16]:
    for train_adversary in ['ID', 'FGSM', 'PGD']:
        config = copy.deepcopy(default_config)
        config['trainer']['max_train_steps'] = 100_000
        config['trainer']['model']['size'] = scale
        config['trainer']['train_adversary'] = {
            '': train_adversary,
            **config['adversaries'][train_adversary],
        }
        config['eval']['final_metrics']['adversaries'] = [
            config['adversaries']['ID'],
            config['adversaries']['FGSM'],
            config['adversaries']['PGD'],
        ]
        del config['adversaries']
        config['eval']['metrics'] = ['loss', 'accuracy']
        log.info('-'*20)
        log.info('NEW RUN')
        log.info('-'*20)
        log.info('\n'+pprint.pformat(config))
        log.info('---')
        results = trainer.eval(config)
        for result in results:
            result['scale'] = scale

        log.info(results)
        df = pd.DataFrame(results)
        acc_df = pd.concat((acc_df, df), ignore_index=True)
        log.info(acc_df)

        res = acc_df
        torch.save(res, respath)
        log.info(f'Saved to {respath}.')



