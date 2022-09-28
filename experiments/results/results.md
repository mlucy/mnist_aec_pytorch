# Overview

All results are compared to https://arxiv.org/pdf/1706.06083.pdf

I reproduced the graphs in section 5, Experiments, for MNIST but not
for CIFAR.  It would potentially be interesting to reproduce figures 1
and 2 from the earlier sections as well.

Their adversarial training scheme is super slow (40 steps of PGD per
outer step of training), so I didn't reproduce the CIFAR results yet.




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

I'm not sure what their "Targeted" attack is, and I had to guess some
of the details for the CW/CW+ attacks.  I haven't looked into model B
yet.  Otherwise very similar.

# Experiment 4 (Figure 6):

## Theirs:

![](exp4_theirs.png)

## Ours:

![](exp4.png)

## Thoughts

I haven't implemented the decision boundary attack yet.  Otherwise
very similar, the brown line in the first graph bottoms out faster for
them.
