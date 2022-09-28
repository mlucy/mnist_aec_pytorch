## TODO
* Implement Targeted, CW, and CW+ adversaries for exp3
* exp4

* https://arxiv.org/pdf/1608.04644.pdf discretizes pixel values to be
  in [0, 256) and then does lattice search if that isn't good enough.
  This paper seems to just let us pass arbitrary floats.  Try it both ways?



## Thoughts

* Because of the way volume in high dimensions works, our random
  starts are usually very close to the surface of the n-cube or n-ball
  that bounds them.  I wonder what would happen if we made the
  distance from the center uniform?

* The paper hypothesizes that the L2-norm PGD attack performs poorly
  on MNIST because the thresholding behavior of the first filters
  stops good gradient information from getting through.  I wonder how
  this would behave if we patched the ReLU units to pass through
  gradients as if they were softmax (just lie basically)?

* I wonder how things behave if instead of training against adversarial attacks within an epsilon ball, the value of epsilon is sampled at training time from some distribution?  Would that get similar results without the sharp dropoff past that epsilon ball?

* The only scaling experiments look at width, I wonder how depth
  affects things?
