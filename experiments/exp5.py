import pandas as pd
import copy
import pprint
import torch
import os
import re

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import utils
utils.log_setup()
log = utils.get_logger(__name__)

import trainer

os.makedirs('results', exist_ok=True)
respath = 'results/exp5.df'

all_results = []
for adam_train in [False, True]:
    config = utils.load_config('default')
    if adam_train:
        config['trainer']['train_adversary'] = {
            '': 'PGD',
            **config['adversaries']['PGD'],
            'name': 'ADAM_PGD',
            'a': 0.3,
            'step_mode': 'adam',
        }
    else:
        config['trainer']['train_adversary'] = {
            '': 'PGD',
            **config['adversaries']['PGD'],
        }

    # config['other_trainers'] = {
    #     "A'": {
    #         **copy.deepcopy(config['trainer']),
    #         'run': 'aprime', # Force us to use a different model.
    #     }
    # }

    _adversaries = [
        {**config['adversaries']['PGD'], 'k': 40, 'restarts': 1, 'random_start': True},
    #    {**config['adversaries']['PGD'], 'k': 100, 'restarts': 1},
    #    {**config['adversaries']['PGD'], 'k': 40, 'restarts': 20},
    #    {**config['adversaries']['PGD'], 'k': 100, 'restarts': 20},

    #    {**config['adversaries']['PGD'], 'k': 40, 'restarts': 1,
    #     'name': 'Adam', 'step_mode': 'adam'},
    #    {**config['adversaries']['PGD'], 'k': 100, 'restarts': 1,
    #     'step_mode': 'adam'},
    #    {**config['adversaries']['PGD'], 'k': 40, 'restarts': 20,
    #     'step_mode': 'adam'},
    #    {**config['adversaries']['PGD'], 'k': 100, 'restarts': 20,
    #     'step_mode': 'adam'},
    ]
    adversaries = []
    for restarts in [1]:
        for steps in [40, 100]:
            for a in [1, 0.3, 0.1, 0.03, 0.01, 0.003]:
                for adv in _adversaries:
                    adversaries.append({**adv, 'k': steps, 'restarts': restarts, 'a': a})
                    adversaries.append({ **adv, 'k': steps, 'restarts': restarts,
                                         'a': a, 'step_mode': 'adam'})

    config['eval']['final_metrics'] = []
    for adversary in adversaries:
        config['eval']['final_metrics'].append(
            {'adversaries': [adversary], 'metrics': ['accuracy']}
        )
    del config['adversaries']

    log.info('-'*20)
    log.info('NEW RUN')
    log.info('-'*20)
    log.info('\n'+pprint.pformat(config))
    log.info('---')
    results = trainer.eval(config)
    for result, conf in zip(results, adversaries):
        result['steps'] = conf.get('k', '-')
        result['restarts'] = conf.get('restarts', '-')
        result['source'] = conf.get('model', 'A')
        result['step_mode'] = conf.get('step_mode', 'paper')
        result['step_size'] = conf.get('a', 0.01)
        del result['step']
        del result['epoch']
        del result['source']
        for k in list(result):
            if m := re.fullmatch('(.+)_accuracy', k):
                # result['name'] = m[1]
                result['accuracy'] = result.pop(k)
    all_results = all_results + results

print(all_results)

log.info(all_results)
df = pd.DataFrame(all_results)
log.info(df)

res = df
torch.save(res, respath)
log.info(f'Saved to {respath}.')



