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
respath = 'results/exp3.df'

config = utils.load_config('default')
config['trainer']['train_adversary'] = {
    '': 'PGD',
    **config['adversaries']['PGD'],
}
config['other_trainers'] = {
    "A'": {
        **copy.deepcopy(config['trainer']),
        'run': 'aprime', # Force us to use a different model.
    }
}

config['eval']['final_metrics']['adversaries'] = [
    {'': 'ID', 'theirs': .988},
    {**config['adversaries']['FGSM'], 'theirs': .956},
    {**config['adversaries']['PGD'], 'k': 40, 'restarts': 1, 'theirs': .932},
    # TODO: The paper is unclear whether we use the same step size
    # for the k=100 cases.  For experiment 4 the paper says to use
    # `2.5 \epsilon / 100`, but it doesn't say what's done here.
    {**config['adversaries']['PGD'], 'k': 100, 'restarts': 1, 'theirs': .918},
    {**config['adversaries']['PGD'], 'k': 40, 'restarts': 20, 'theirs': .904},
    {**config['adversaries']['PGD'], 'k': 100, 'restarts': 20, 'theirs': .893},
    # TODO: What the fuck is this "targeted" row?  I might just be
    # blind but I can't find it defined anywhere.  Maybe it's just
    # CW with a specific target class?

    # Targeted 40 1 A
    {**config['adversaries']['PGD'], '': 'CW', 'name': 'CW', 'theirs': .940},
    {**config['adversaries']['PGD'],
     '': 'CW', 'k': 50, 'name': 'CW+', 'theirs': .939},

    {**config['adversaries']['FGSM'], 'model': "A'", 'theirs': .968},
    {**config['adversaries']['PGD'], 'model': "A'",
     'k': 40, 'restarts': 1, 'theirs': .960},
    {**config['adversaries']['PGD'], 'model': "A'",
     'k': 100, 'restarts': 20, 'theirs': .957},
    {**config['adversaries']['PGD'], 'model': "A'",
     '': 'CW', 'name': 'CW', 'theirs': .970},
    {**config['adversaries']['PGD'], 'model': "A'",
     '': 'CW', 'k': 50, 'name': 'CW+', 'theirs': .964},

    # FGSM B,
    # PGD 40 1 B,
    # CW+ B
]
theirs = []
for i, o in enumerate(config['eval']['final_metrics']['adversaries']):
    o['name'] = f'{i}'
    theirs.append(o['theirs'])
    del o['theirs']
del config['adversaries']
log.info('-'*20)
log.info('NEW RUN')
log.info('-'*20)
log.info('\n'+pprint.pformat(config))
log.info('---')
_results = trainer.eval(config)
print(_results)
results = []
for _result in _results:
    for i, o in enumerate(config['eval']['final_metrics']['adversaries']):
        to_append = copy.deepcopy(o)
        o['adversary'] = o['']
        del o['']
        results.append({
            **o,
            'accuracy': _result[f'{i}_accuracy'],
            'theirs': theirs[i],
        })

log.info(results)
df = pd.DataFrame(results)
log.info(df)

res = df
torch.save(res, respath)
log.info(f'Saved to {respath}.')



