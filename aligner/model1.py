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

''' Add null character to the translated text. '''
def add_null(bitext_raw):
    sys.stderr.write('Adding null character to the translated text.\n')
    bitext = []
    # add in the null character
    for (f,e) in bitext_raw:
        e.append('')
        bitext.append([f,e])

    return bitext

''' Trains the model on the corpus.
    Params:
        bitext      Corpus to train on.
        iterations  Number of times to iterate for EM. '''
def train_model(bitext, iterations):
    sys.stderr.write('Training model...\n')
    e_count = set()
    f_count = set()
    f_prob = defaultdict(Decimal)
    s_total = defaultdict(Decimal)

    uniform_t = 0
    t = defaultdict(Decimal)

    sys.stderr.write('\tCounting f,e in bitext...\n')
    sys.stderr.write('\tInitializing t to 0...\n')
    sys.stderr.write('\t\t')
    for (n, (f, e)) in enumerate(bitext):
        sys.stderr.write('.')
        for f_i in f:
            f_count.add(f_i)
            for e_i in e:
                e_count.add(e_i)
                t[(e_i, f_i)] = 0

    # default value provided as uniform probability)
    uniform_t = Decimal(1/len(f_count))

    # init to uniform prob
    sys.stderr.write('\tSetting t to uniform probability...\n')
    for fe in t:
        t[fe] = uniform_t

    # init loop
    # see koehn's pseudocode
    sys.stderr.write('\tStarting EM iterations...\n')
    for i in range(iterations):
        sys.stderr.write("\n\t\tBeginning iteration: %i\n" % i)
        count = defaultdict(Decimal)
        total = defaultdict(Decimal)
        for (n, (f, e)) in enumerate(bitext):
            sys.stderr.write("\t\t\tCalculating for line: %i\n" % n)
            for e_i in e:
                s_total[e_i] = 0
                for f_i in f:
                    s_total[e_i] += t[(e_i,f_i)]

            for e_i in e:
                for f_i in f:
                    count[(e_i,f_i)] += t[(e_i,f_i)] / s_total[e_i]
                    total[f_i] += t[(e_i,f_i)] / s_total[e_i]

        sys.stderr.write('\t\tRecalculating translation probability...\n')
        for f_i in f_count:
            sys.stderr.write('.')
            for e_i in e_count:
                t[(e_i,f_i)] = count[(e_i,f_i)] / total[f_i]

    return t

''' Aligns all the phrases in a given bitext corpus.
    Params:
        t   translation probability'''
def align(t, bitext):
    # alignment
    sys.stderr.write('Beginning alignment...\n')
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
    bitext_raw = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

    bitext = add_null(bitext_raw)
    t = train_model(bitext, opts.iterations)
    align(t, bitext)
