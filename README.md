## TODO

* Add a requirements file.

* Try Adam for PGD update.
  - Check how often we actually get to the shell.
  - Seems like behavior is bad if we try to step past the shell
    multiple times and learn an unreasonable level of certainty.

* Try fast FGSM: https://arxiv.org/abs/2001.03994

* Try early stopping: https://arxiv.org/pdf/2002.11569.pdf

* Add lock-holding for the training cache so experiments can be run in
  parallel without re-doing work?

## Thoughts

* https://arxiv.org/pdf/1608.04644.pdf discretizes pixel values to be
  in [0, 256) and then does lattice search if that isn't good enough.
  This paper seems to just let us pass arbitrary floats.  Try it both ways?

* Because of the way volume in high dimensions works, our random
  starts are usually very close to the surface of the n-cube or n-ball
  that bounds them.  I wonder what would happen if we made the
  distance from the center uniform?

* The paper hypothesizes that the L2-norm PGD attack performs poorly
  on MNIST because the thresholding behavior of the first filters
  stops good gradient information from getting through.  I wonder how
  this would behave if we patched the ReLU units to pass through
  gradients as if they were softmax (just lie basically)?

* I wonder how things behave if instead of training against
  adversarial attacks within an epsilon ball, the value of epsilon is
  sampled at training time from some distribution?  Would that get
  similar results without the sharp dropoff past that epsilon ball?

* The only scaling experiments look at width, I wonder how depth
  affects things?
