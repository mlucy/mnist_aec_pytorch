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

adversaries = [
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

    # TODO: I don't know what confidence to put for the CW adversary.
    # The paper doesn't say, and 0 gives results that don't match
    # theirs (while 50 matches their CW+ quite well).  I went with 20
    # because it seems to give roughly the right numbers and the CW
    # paper mentions it as the success plateaux for transfer attacks
    # (pg. 15).
    {**config['adversaries']['PGD'], '': 'CW', 'name': 'CW',
     'confidence': 20, 'restarts': 1, 'theirs': .940},
    {**config['adversaries']['PGD'], '': 'CW', 'name': 'CW+',
     'confidence': 50, 'restarts': 1, 'theirs': .939},

    {**config['adversaries']['FGSM'], 'model': "A'", 'theirs': .968},
    {**config['adversaries']['PGD'], 'model': "A'",
     'k': 40, 'restarts': 1, 'theirs': .960},
    {**config['adversaries']['PGD'], 'model': "A'",
     'k': 100, 'restarts': 20, 'theirs': .957},
    {**config['adversaries']['PGD'], 'model': "A'",
     '': 'CW', 'name': 'CW', 'confidence': 20, 'restarts': 1, 'theirs': .970},
    {**config['adversaries']['PGD'], 'model': "A'",
     '': 'CW', 'confidence': 50, 'restarts': 1, 'name': 'CW+', 'theirs': .964},

    # FGSM B,
    # PGD 40 1 B,
    # CW+ B
]

theirs = []
config['eval']['final_metrics'] = []
for adversary in adversaries:
    theirs.append(adversary.pop('theirs'))
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
for result, x, conf in zip(results, theirs, adversaries):
    result['theirs'] = x
    result['steps'] = conf.get('k', '-')
    result['restarts'] = conf.get('restarts', '-')
    result['source'] = conf.get('model', 'A')
    for k in list(result):
        if m := re.fullmatch('(.+)_accuracy', k):
            result['name'] = m[1]
            result['accuracy'] = result.pop(k)
print(results)

log.info(results)
df = pd.DataFrame(results)
log.info(df)

res = df
torch.save(res, respath)
log.info(f'Saved to {respath}.')



