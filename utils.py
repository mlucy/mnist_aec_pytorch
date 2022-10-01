import copy
import json5 as json
import logging
import os
import pathlib
import pprint
import re
import torch
import uuid

def log_setup():
    formatter = logging.Formatter(
        '{levelname[0]} {message:<60s} | ({name})',
        style='{')

    fh = logging.FileHandler('debug.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    ch.setFormatter(formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=[fh, ch])
    root_logger = logging.getLogger('')
    root_logger.addHandler(ch)
    root_logger.addHandler(fh)

    return root_logger

def get_logger(name):
    return logging.getLogger(name)

log = get_logger(__name__)

def iter_params(obj, f):
    for m in obj.children():
        iter_params(m, f)
    for p in obj._parameters.values():
        f(obj, pn, p)

def log_cfg():
    log.info(pprint.pformat(CFG))

def load_config(name):
    cur_dir = str(pathlib.Path(__file__).parent.resolve())
    fpath = os.path.join(cur_dir, 'configs', f'{name}.json')
    with open(fpath, 'r') as f:
        return json.load(f)

def json_hash(obj):
    encoded = json.dumps(obj, sort_keys=True)
    return str(uuid.uuid5(uuid.NAMESPACE_OID, encoded))

def fs_cacheable(path='.cache', version='v2'):
    def __f(f):
        def _f(*a, **kw):
            os.makedirs(path, exist_ok=True)
            cache_str = json_hash({'a': a, 'kw': kw})
            cache_path = os.path.join(
                path,
                f'{f.__name__}-{version}-{cache_str}.pt',
            )
            if os.path.exists(cache_path):
                log.info(f'{cache_path} exists, loading.')
                return torch.load(cache_path)
            else:
                log.info(f'{cache_path} does not exist, computing.')
                res = f(*a, **kw)
                log.info(f'Saving to {cache_path}.')
                torch.save(res, cache_path)
                return res
        return _f
    return __f

class Config:
    def __init__(self, config):
        defaults = {}
        required = set([])
        allowed = set([])
        for cls in type.mro(type(self))[-2::-1]:
            defaults = {**defaults, **getattr(cls, '_defaults', {})}
            required = required.union(getattr(cls, '_required', set([])))
            allowed = allowed.union(getattr(cls, '_allowed', set([])))

        self._obj = {**defaults, **config}

        for k in self._obj:
            if (k not in defaults and
                k not in required and
                k not in allowed):
                raise RuntimeError(f'unrecognized config {k}')
        for k in required:
            if k not in self._obj:
                raise RuntimeError(f'missing required config {k}')

    def __repr__(self):
        return f'Config({self._obj.__repr__()})'

    def __getattr__(self, field):
        if field in self._obj:
            return self._obj[field]
        else:
            raise AttributeError(f'no config field {field}')

    def __getstate__(self):
        return self._obj

    def __setstate__(self, state):
        self._obj = state

class Writable:
    def __init__(self, unwritable):
        self._unwritable = unwritable
        for k, f in self._unwritable.items():
            setattr(self, k, f())

    def __getstate__(self):
        state = copy.copy(self.__dict__)
        for k in self._unwritable:
            del state[k]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for k, f in self._unwritable.items():
            setattr(self, k, f())
