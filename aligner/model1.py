#!/usr/bin/env python
from __future__ import division
import optparse
import sys
import collections
import decimal
from collections import defaultdict
from decimal import Decimal

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-i", "--iterations", dest="iterations", default=10, type="int", help="Number of times to iterate over the text.")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

# set precision of all decimals
decimal.getcontext().prec = 4
decimal.getcontext().rounding = decimal.ROUND_HALF_UP

# Contains an array of all sentences in the text
# Each element is a 2-element array, [0] is the french translation of the sentence, and [1] is the english translation.
bitext_raw = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

bitext = []

def add_null(bitext_raw):
    bitext = []
    # add in the null character
    for (f,e) in bitext_raw:
        e.append('')
        bitext.append([f,e])

    return bitext

''' Trains the model on the corpus.
    Params:
        bitext      Corpus to train on.'''
def train_model(bitext):
    e_count = set()
    f_count = set()
    f_prob = defaultdict(Decimal)
    s_total = defaultdict(Decimal)

    uniform_t = 0
    t = defaultdict(Decimal)

    for (n, (f, e)) in enumerate(bitext):
        # TODO: Remove punctuation?

        for f_i in f:
            f_count.add(f_i)
            for e_i in e:
                e_count.add(e_i)
                t[(e_i, f_i)] = 0

    # default value provided as uniform probability)
    uniform_t = Decimal(1/len(f_count))

    # init to uniform prob
    for fe in t:
        t[fe] = uniform_t

    # init loop
    # see koehn's pseudocode
    for i in range(opts.iterations):
        count = defaultdict(Decimal)
        total = defaultdict(Decimal)
        for (n, (f, e)) in enumerate(bitext):
            for e_i in e:
                s_total[e_i] = 0
                for f_i in f:
                    s_total[e_i] += t[(e_i,f_i)]

            for e_i in e:
                for f_i in f:
                    count[(e_i,f_i)] += t[(e_i,f_i)] / s_total[e_i]
                    total[f_i] += t[(e_i,f_i)] / s_total[e_i]

        for f_i in f_count:
            for e_i in e_count:
                t[(e_i,f_i)] = count[(e_i,f_i)] / total[f_i]

    return t

''' Aligns all the phrases in a given bitext corpus.
    Params:
        t   translation probability'''
def align(t):
    # alignment
    for (f, e) in bitext:
        for (i, f_i) in enumerate(f):
            max_t = 0
            max_align = 0
            for (j, e_j) in enumerate(e):
                # choose highest probability of all alignments
                if max_t < t[(e_j,f_i)]:
                    max_t = t[(e_j,f_i)]
                    max_align = j
            sys.stdout.write("%i-%i " % (i,max_align))
        sys.stdout.write("\n")

if __name__ == '__main__':
    bitext = add_null(bitext_raw)
    t = train_model(bitext)
    align(t)
