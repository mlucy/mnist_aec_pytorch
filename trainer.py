from torch import nn
from torch.utils.tensorboard import SummaryWriter
import copy
import os
import torch
import torch.nn.functional as F
import numpy as np
import time

import utils
log = utils.get_logger(__name__)
from dataset import mnist_dataloader
import model
import adversary

class TrainerConfig(utils.Config):
    _defaults = {
        'base_dir': '.',
        'output_steps': 100,
        'summary_steps': 100,
        'batch_sz': 50,
        'eval_batch_sz': 200,
        'max_train_steps': 100_000,
        'retain_test_metrics': False,
        'retain_train_metrics': False,
        'run': None,
    }

class Trainer(utils.Writable):
    def __init__(self, _config):
        config = copy.deepcopy(_config)

        self.model = self.subconfig(**config.pop('model')).to('cuda')
        self.opt = self.subconfig(params=self.model.parameters(), **config.pop('opt'))
        self.train_adversary = self.subconfig(**config.pop('train_adversary'))

        if config.get('test_adversaries', None) is None:
            self.test_adversaries = [self.train_adversary]
        else:
            self.test_adversaries = [
                self.subconfig(**c)
                for c in config.pop('test_adversaries')
            ]

        self.config = TrainerConfig(config)

        self.step = 0
        self.epoch = 0

        self.train_metrics_acc = []
        self.train_metrics = []

        super().__init__({
            'writer': self.construct_writer,
        })

    def construct_writer(self):
        c = self.config
        if c.run is None:
            return None
        return SummaryWriter(os.path.join(c.base_dir, 'runs', c.run))

    @classmethod
    def subconfig(_, **subconfig):
        CLSMAP = {
            'ADAM': torch.optim.Adam,
            'MNIST': model.Mnist,
            'FGSM': adversary.FGSM,
            'PGD': adversary.PGD,
            'ID': adversary.ID,
        }
        cls = subconfig.pop('')
        return CLSMAP[cls](**subconfig)

    def train_one_batch(self, xs, ys):
        logits = self.model(xs)
        loss = self.model.loss(logits, ys)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {
            'loss': loss.item(),
            'accuracy': self.model.accuracy(logits, ys).item(),
        }

    def log(self, ns, metrics):
        c = self.config
        for k, v in metrics.items():
            if self.writer is not None:
                self.writer.add_scalar(f'{k}/{ns}', v, self.step)
        if self.step % c.output_steps == 0:
            prev_time = getattr(self, '_start_time', None)
            self._start_time = time.time()
            if prev_time is not None:
                batch_per_sec = c.output_steps / (self._start_time - prev_time)
                out_str = f'{ns} {self.step}:'
                for k, v in metrics.items():
                    out_str += f' {k}={v:.03f}'
                out_str += f' ({batch_per_sec})'
                log.info(out_str)

    def grad_cb(self, ys, loss_fn=None):
        def f(points):
            # TODO: should we be explicitly setting train to True or False
            # here?  Probably it's fine either way.
            with torch.enable_grad():
                points_param = nn.Parameter(points.detach())
                assert points_param.grad is None
                logits = self.model(points_param)
                if loss_fn is None:
                    loss = self.model.loss(logits, ys)
                else:
                    loss = loss_fn(logits, ys)
                loss.backward()
                return (
                    points_param.grad.detach(),
                    self.model.loss(logits, ys, reduce=False).detach(),
                )
        return f

    def train_one_epoch(self):
        c = self.config
        self._start_time = None
        log.info(f'Training for epoch {self.epoch}...')
        train_dataloader = mnist_dataloader(
            batch_sz=c.batch_sz, train=True, shuffle=True)
        self.model.train(True)
        start_time = None
        for _xs, ys in train_dataloader:
            with torch.no_grad():
                xs = self.train_adversary.perturb(_xs, self.grad_cb(ys))
            res = self.train_one_batch(xs, ys)

            if c.retain_train_metrics:
                self.train_metrics_acc.append(res)
                if len(self.train_metrics_acc) >= c.summary_steps:
                    acc = {
                        f'train_{k}': np.mean([x[k] for x in self.train_metrics_acc])
                        for k in self.train_metrics_acc[0]
                    }
                    new = {
                        'step': self.step,
                        'epoch': self.epoch,
                        'train_adversary': self.train_adversary.config.name,
                        **acc,
                    }
                    self.train_metrics.append(new)
                    self.train_metrics_acc = []

            self.log('train', res)
            self.step += 1
            if self.step >= c.max_train_steps:
                return False
        log.info(f'Training for epoch {self.epoch} completed.')
        self.epoch += 1
        return True

    def _test(self, adversary, adversary_trainer=None):
        log.info('Evaluating.')
        c = self.config
        test_dataloader = mnist_dataloader(
            c.eval_batch_sz, train=False, shuffle=False)
        metrics = {'loss': 0, 'accuracy': 0}
        n = 0
        self.model.train(False)
        if adversary_trainer is None:
            adversary_trainer = self
        else:
            adversary_trainer.model.train(False)
        with torch.no_grad():
            for _xs, ys in test_dataloader:
                xs = adversary.perturb(_xs, adversary_trainer.grad_cb(ys))
                logits = self.model(xs)
                metrics['loss'] += self.model.loss(logits, ys).item()
                metrics['accuracy'] += self.model.accuracy(logits, ys).item()
                n += 1
        for k, v in metrics.items():
            metrics[k] = v/n
        return metrics

    def test(self):
        c = self.config
        for adversary in self.test_adversaries:
            metrics = self._test(adversary)
            if c.retain_test_metrics:
                test_metrics = self.__dict__.setdefault('test_metrics', [])
                test_metrics.append({
                    'step': self.step,
                    'epoch': self.epoch,
                    'train_adversary': self.train_adversary.config.name,
                    'test_adversary': adversary.config.name,
                    **metrics
                })
            self.log(f'{adversary.config.name}/test', metrics)

    def train(self):
        while True:
            self.test()
            more = self.train_one_epoch()
            if not more:
                log.info('Training complete.')
                break

@utils.fs_cacheable()
def train(train_config):
    trainer = Trainer(train_config)
    trainer.train()
    return trainer

def train_test_metrics(_train_config):
    train_config = copy.deepcopy(_train_config)
    train_config['retain_train_metrics'] = True
    train_config['retain_test_metrics'] = True
    trainer = train(train_config)
    return trainer.train_metrics, trainer.test_metrics

@utils.fs_cacheable()
def eval(config):
    trainer = train(config['trainer'])
    data_points = []
    other_trainers = {}
    for k, v in config.get('other_trainers', {}).items():
        other_trainers[k] = train(v)
    final_metrics = config['eval']['final_metrics']
    if isinstance(final_metrics, dict):
        final_metrics = [final_metrics]
    for final_metric in final_metrics:
        obj = {
            'step': trainer.step,
            'epoch': trainer.epoch,
            'train_adversary': trainer.train_adversary.config.name,
            **final_metric.get('tags', {}),
        }
        for adversary_config in final_metric['adversaries']:
            adversary = Trainer.subconfig(**adversary_config)
            adversary_trainer = trainer
            if getattr(adversary.config, 'model', None) is not None:
                adversary_trainer = other_trainers[adversary.config.model]
            log.info(f'Evaluating on {adversary_config}.')
            results = trainer._test(adversary, adversary_trainer)
            patch = {
                f'{adversary.config.name}_{k}': results[k]
                for k in final_metric['metrics']
            }
            obj = {**obj, **patch}
        data_points.append(obj)
    return data_points
