import torch
from torch import nn
from torch.nn import functional as F
from utils import Config

class MnistConfig(Config):
    _defaults = {
        'dim': 28,
        'size': 16,
        'kernel_size': 5,
        'in_channels': 1,
        'scale_factor': 2,
        'hidden_layer_size': 64,
    }

class Mnist(nn.Module):
    def __init__(self, **config):
        super().__init__()
        c = self.config = MnistConfig(config)
        self.conv1 = nn.Conv2d(c.in_channels, c.in_channels*c.scale_factor*c.size,
                               kernel_size=c.kernel_size, padding='same')
        self.conv2 = nn.Conv2d(self.conv1.out_channels,
                               self.conv1.out_channels*c.scale_factor,
                               kernel_size=c.kernel_size, padding='same')
        final_acts = (c.dim // (c.scale_factor**2))**2 * self.conv2.out_channels
        self.fc1 = nn.Linear(final_acts, c.hidden_layer_size*c.size)
        self.fc2 = nn.Linear(c.hidden_layer_size*c.size, 10)

        # https://github.com/MadryLab/mnist_challenge/blob/master/model.py#L58
        # Not in the paper, probably not important, but copied it anyway.
        def init(obj, param_name, param):
            if re.fullmatch(param_name, '.*bias'):
                nn.init.constant_(param, 0.1)
            elif re.fullmatch(param_name, '.*weight'):
                nn.init.trunc_normal_(stddev=0.1, a=-0.2, b=0.2)

    @classmethod
    def loss(_, logits, y, reduce=None):
        # TODO: I think they might be using a `sum` reduction for
        # training even though they use an `avg` reduction for the
        # reported numbers in the paper?  That might explain some of
        # the minor numerical differences, although with Adam as the
        # optimizer it shouldn't be super sensitive to the scale of
        # the loss.
        return F.cross_entropy(logits, y, reduce=reduce)

    @classmethod
    def accuracy(_, logits, y):
        preds = torch.argmax(logits, dim=-1)
        return torch.mean((preds == y).to(torch.float32))

    def forward(self, _batch):
        c = self.config
        batch = _batch
        batch_sz = batch.shape[0]
        assert batch.shape == (batch_sz, 1, c.dim, c.dim)

        batch = F.relu(self.conv1(batch))
        batch = F.max_pool2d(batch, 2)
        batch = F.relu(self.conv2(batch))
        batch = F.max_pool2d(batch, 2)

        batch = batch.reshape(batch_sz, -1)
        batch = F.relu(self.fc1(batch))
        logits = self.fc2(batch)

        return logits
