# Overview

All results are compared to https://arxiv.org/pdf/1706.06083.pdf

I reproduced the graphs in Section 5 (Experiments) for MNIST.  It
would potentially be interesting to reproduce figures 1 and 2 from the
earlier sections as well.

Their adversarial training scheme is super slow (40 steps of PGD per
outer step of training), so I didn't reproduce the CIFAR results yet.

To reproduce e.g. experiment 1:
```
cd mnist_aec_pytorch/experiments
python exp1.py
python exp1_plot.py results/exp1.df
open results/exp1.png
```


# Experiment 1 (Figure 4)

## Theirs:

![](exp1_theirs.png)

## Ours:

![](exp1.png)

## Thoughts

Overall these look very similar.  Our capacity 4/8 seems to be
behaving slightly differently, not sure why, should probably re-run
the training and see if it was just an outlier.

# Experiment 2 (Figure 5)

## Theirs:

![](exp2_theirs.png)

## Ours:

![](exp2.png)

## Thoughts

Overall these look very similar.  We seem to bottom out and start
rising in loss earlier than they do and at a slightly higher loss.
That seems suspicious to me, I should double-check the training code.

# Experiment 3 (Table 1)

## Theirs:

![](exp3_theirs.png)

## Ours:

![](exp3.png)

## Thoughts

Everything mostly matches up, our accuracy numbers seem very slightly
higher.

TODO:

* I don't understand the attack they labeled "Targeted", I can't seem
  to find where they describe it in the paper.
* I don't know what to set the convidence parameter to for CW (for CW+
  they specify 50).  I ended up going with 20 (see comments in
  exp3.py).
* Why are our accuracy numbers consistently a little higher?  Seems
  odd, seems true over multiple runs.  Might be nothing.

# Experiment 4 (Figure 6):

## Theirs:

![](exp4_theirs.png)

## Ours:

![](exp4.png)

## Thoughts

I haven't implemented the decision boundary attack yet.  Otherwise
very similar, the brown line in the first graph bottoms out faster for
them.
