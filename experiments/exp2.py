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
respath = 'results/exp2.df'

config = utils.load_config('default')
config['trainer']['train_adversary'] = config['adversaries']['PGD']
config['trainer']['retain_test_metrics'] = True
del config['adversaries']
log.info('-'*20)
log.info('NEW RUN')
log.info('-'*20)
log.info('\n'+pprint.pformat(config['trainer']))
log.info('---')
train_metrics, test_metrics = trainer.train_test_metrics(config['trainer'])
res = {
    'train': pd.DataFrame(train_metrics),
    'test': pd.DataFrame(test_metrics),
}

log.info(res)

torch.save(res, respath)
log.info(f'Saved to {respath}.')



