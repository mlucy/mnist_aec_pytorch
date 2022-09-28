import utils
import torch
import numpy as np

class LinfNorm:
    def __init__(self, config, start_point):
        c = self.config = config
        self.start_point = start_point

        # We want to make sure we never produce an invalid image.
        # Note: pytorch is insane, so the second argument has to be
        # explicitly cast to a tensor, or it will be interpreted as a
        # dimension index.
        self.minval = torch.max(self.start_point - c.eps, torch.tensor(0))
        self.maxval = torch.min(self.start_point + c.eps, torch.tensor(1))

    def random_uniform(self):
        c = self.config
        noise = (torch.rand_like(self.start_point) * c.eps * 2) - c.eps
        return self.start_point + noise

    def step(self, point, grad):
        c = self.config
        return self.project(point + c.a * torch.sign(grad))

    def project(self, point):
        return point.clip(min=self.minval, max=self.maxval)

    def max_shell(self, point, grad):
        c = self.config
        return self.project(point + 2*c.eps*torch.sign(grad))

class L2Norm:
    def __init__(self, config, start_point):
        c = self.config = config
        self.start_point = start_point

        # We want to make sure we never produce an invalid image.
        # Note: pytorch is insane, so the second argument has to be
        # explicitly cast to a tensor, or it will be interpreted as a
        # dimension index.
        self.minval = torch.zeros_like(start_point)
        self.maxval = torch.ones_like(start_point)

        self.shape = (1, 28, 28)

    def vector_norm(self, vec):
        # Unlike the functions for the Linf norm, we need to care
        # about which dimensions are our image dimensions.
        assert vec.shape[-3:] == self.shape
        norm = torch.linalg.norm(vec, dim=[-3, -2, -1])
        assert norm.shape == vec.shape[:-3]
        norm = norm[..., None, None, None]
        return norm + 1e-20

    def random_uniform(self):
        c = self.config

        # TODO: Check that this math is correct.  Muller method I
        # think it's called?

        # We're trying to sample from the volume of an n-ball (I
        # think, the paper is unclear and I could only find their code
        # for the Linf norm).
        noise = torch.randn_like(self.start_point)
        noise /= self.vector_norm(noise)
        noise *= c.eps
        r = torch.rand((), device=self.start_point.device)
        r = r ** (1/np.prod(self.start_point.shape[-3:]))
        return self.start_point + r * noise

    def step(self, point, grad):
        c = self.config
        new_point = point + c.a * grad / self.vector_norm(grad)
        return self.project(new_point)

    def project(self, point):
        c = self.config
        vec = (point - self.start_point)
        norm = self.vector_norm(vec)
        new_norm = norm.clip(max=c.eps)
        new_point = self.start_point + vec * new_norm / norm
        res = new_point.clip(min=self.minval, max=self.maxval)
        assert not res.isnan().any()
        return res

    def max_shell(self, point, grad):
        c = self.config
        return self.project(point + 2*c.eps*self.grad/self.vector_norm(grad))

class Adversary:
    def make_norm(self, start_point):
        c = self.config
        cls = {
            'linf': LinfNorm,
            'l2': L2Norm,
        }[c.norm]
        return cls(c, start_point)

class IDConfig(utils.Config):
    _defaults = {
        'name': 'ID',
    }
class ID(Adversary):
    def __init__(self, **config):
        self.config = IDConfig(config)

    def perturb(self, start_point, grad_cb):
        return start_point

class FGSMConfig(utils.Config):
    _defaults = {
        'name': 'FGSM',
        'eps': 0.3,
        'random_start': False,
        'norm': 'linf',
        'model': None,
    }
class FGSM(Adversary):
    def __init__(self, **config):
        self.config = FGSMConfig(config)

    def perturb(self, start_point, grad_cb):
        c = self.config
        norm = self.make_norm(start_point)
        if c.random_start:
            cur = norm.random_uniform()
        else:
            cur = start_point

        grad, _loss = grad_cb(cur)
        return norm.max_shell(cur, grad)

class PGDConfig(utils.Config):
    _defaults = {
        'name': 'PGD',
        'eps': 0.3,
        'k': 40,
        'a': 0.01,
        'random_start': True,
        'restarts': 1,
        'norm': 'linf',
        'model': None,
    }
class PGD(Adversary):
    def __init__(self, **config):
        self.config = PGDConfig(config)

    def perturb(self, start_point, grad_cb):
        c = self.config
        best_loss = start_point.new_full(start_point.shape[:1], -torch.inf)
        best = torch.zeros_like(start_point)
        for _ in range(c.restarts):
            norm = self.make_norm(start_point)
            if c.random_start:
                cur = norm.random_uniform()
            else:
                cur = start_point

            for i in range(c.k):
                grad, _loss = grad_cb(cur)
                cur = norm.step(cur, grad)
            _grad, loss = grad_cb(cur)
            assert loss.shape == best_loss.shape
            # We do it in this order so if k=1 and loss is NaN or -inf
            # we still return the perturbed point instead of zeros.
            best = torch.where(
                (best_loss > loss)[:, None, None, None],
                best,
                cur,
            )
        return best

class CWConfig(PGDConfig):
    _defaults = {
        'confidence': 0,
    }
class CW(PGD):
    def __init__(self, **config):
        self.config = CWConfig(config)

    # Adapted from pg. 10 of https://arxiv.org/pdf/1608.04644.pdf.
    # The main difference is that A) We're doing PGD instead of trying
    # to find the smallest attack that works, and B) We're trying to
    # get any wrong answer rather than a specific target, so the max
    # is flipped.
    def loss_fn(self, logits, ys, reduce=None):
        c = self.config
        batch_sz, n_cls = logits.shape
        arange = torch.arange(n_cls, device=logits.device)
        true_val_mask = arange[None, :].tile((batch_sz, 1)) == ys[:, None]
        # This needs to match the type of `logits` or `where`
        # complains, even if one of the types is promotable.
        ninf = logits.new_full((), -torch.inf)
        max_true_logit = torch.max(
            torch.where(true_val_mask, logits, ninf),
            dim=1,
        ).values
        max_false_logit = torch.max(
            torch.where(true_val_mask.logical_not(), logits, ninf),
            dim=1,
        ).values
        # This step doesn't really make sense for PGD.  Normally the
        # idea is to make the network just-barely-wrong and then
        # minimize distance, but now that we're allowed to be up to
        # epsilon wrong, this just stopes us from optimizing once we
        # reach a known certainty.  Maybe we're supposed to still be
        # minimizing distance?  The paper is unclear.
        loss = torch.max(max_false_logit - max_true_logit,
                         torch.tensor(-c.confidence))
        if reduce is None:
            return torch.mean(loss)
        elif reduce is False:
            return loss
        else:
            raise RuntimeError(f'unsupported loss reduction {reduce}')

    def perturb(self, start_point, grad_cb):
        def new_grad_cb(point):
            return grad_cb(point, loss_fn=self.loss_fn)
        return super().perturb(start_point, new_grad_cb)
