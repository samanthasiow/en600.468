#!/usr/bin/env python
from __future__ import division
import optparse
import sys
import collections
from collections import defaultdict
from decimal import Decimal

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

# Contains an array of all sentences in the text
# Each element is a 2-element array, [0] is the french translation of the sentence, and [1] is the english translation.
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]


e_count = set()
f_count = set()
a = defaultdict(Decimal)
count_ef = defaultdict(Decimal)
total_f = defaultdict(Decimal)
count_a = defaultdict(Decimal)
total_a = defaultdict(Decimal)

s_total = defaultdict(Decimal)

# get t(e|f) from ibm  model 1

#initialize a(i|j,le,lf) = 1/(lf+1) for all i, j, le, lf
for (n, (f, e)) in enumerate(bitext):

    l_e = len(e)
    l_f = len(f)
    for (j, e_i) in enumerate(e, 1): # 0...l_e
        total_a[(j, l_e, l_f)] = 0
        for (i, f_i) in enumerate(f): # 1...l_f
            total_f[f_i] = 0
            total_a[(j, l_e, l_f)] = 0
            count_ef[(e_i, f_i)] = 0
            count_a[(i, j, l_e, l_f))] = 0
            a[(i, j, l_e, l_f)] = D(1)/D(l_f +1) #---???


for (n, (f, e)) in enumerate(bitext):
    # compute normalization
    for (j, e_i) in enumerate(e, 1): # 1...l_e
        s_total[e_i] = 0
        for (i, f_i) in enumerate(f): # 0...l_f
            s_total[e_i] += t[(e_i, f_i)] * a[(i, j, l_e, l_f)]

    # collect counts
    for (j, e_i) in enumerate(e, 1): # 1...l_e
        for (i, f_i) in enumerate(f):  # 0...l_f
            c = t[(e_i, f_i)] * a[(i, j, l_e, l_f)] / s_total[e_i]
            count_ef[(e_i, f_i)] += c
            total_f[f_i] += c
            count_a[(i, j, l_e, l_f))] += c
            total_a[(j, l_e, l_f)] += c

# estimate probabilities
for (n, (f, e)) in enumerate(bitext):
    l_e = len(e)
    l_f = len(f)
    for f_i in f:
        for e_i in e:
            t[(e_i, f_i)] = 0
            a[(e_i, f_i, l_e, l_f)] = 0

for (n, (f, e)) in enumerate(bitext):
    l_e = len(e)
    l_f = len(f)
    for (j, e_i) in enumerate(e, 1):
        for (i, f_i) in enumerate(f):
            t[(e_i, f_i)] = Decimal(count_ef[(e_i, f_i)]) / Decimal(total_f[f_i])
            a[(i, j, l_e, l_f)] = Decimal(count_a[(i, j, l_e, l_f)]) / \
                                      Decimal(total_a[(j, l_e, l_f)])

# return (t, a)
