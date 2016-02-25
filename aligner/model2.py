#!/usr/bin/env python
from __future__ import division
import optparse
import sys
import collections
from collections import defaultdict
import decimal
from decimal import Decimal

import model1 as m1

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-i", "--iterations", dest="iterations", default=10, type="int", help="Number of times to iterate over the text.")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

# Contains an array of all sentences in the text
# Each element is a 2-element array, [0] is the french translation of the sentence, and [1] is the english translation.
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

def train_model(bitext, iterations):
    sys.stderr.write('Training model...\n')
    a = defaultdict(Decimal)
    count_ef = defaultdict(Decimal)
    total_f = defaultdict(Decimal)
    count_a = defaultdict(Decimal)
    total_a = defaultdict(Decimal)

    s_total = defaultdict(Decimal)

    t = m1.train_model(bitext, 1)

    #initialize a(i|j,le,lf) = 1/(lf+1) for all i, j, le, lf
    for (n, (f, e)) in enumerate(bitext):
        l_e = len(e)
        l_f = len(f)
        for (j, e_i) in enumerate(e, 1): # 0...l_e
            total_a[(j, l_e, l_f)] = 0
            for (i, f_i) in enumerate(f, 1): # 1...l_f
                total_f[f_i] = 0
                total_a[(j, l_e, l_f)] = 0
                count_ef[(e_i, f_i)] = 0
                count_a[(i, j, l_e, l_f)] = 0
                a[(i, j, l_e, l_f)] = Decimal(1)/Decimal(l_f + 1) #---???

    sys.stderr.write('\tStarting EM iterations...\n')
    iterations = 1
    for i in range(iterations):
        sys.stderr.write("\t\tBeginning iteration: %i\n" % i)

        sys.stderr.write('\t\tComputing Normalization...\n')
        for (n, (f, e)) in enumerate(bitext):
            l_e = len(e)
            l_f = len(f)
            # compute normalization
            for (j, e_i) in enumerate(e, 1): # 1...l_e
                s_total[e_i] = 0
                for (i, f_i) in enumerate(f, 1): # 0...l_f
                    s_total[e_i] += Decimal(t[(e_i, f_i)]) * Decimal(a[(i, j, l_e, l_f)])



            # collect counts
            for (j, e_i) in enumerate(e, 1): # 1...l_e
                for (i, f_i) in enumerate(f, 1):  # 0...l_f
                    c = t[(e_i, f_i)] * a[(i, j, l_e, l_f)] / s_total[e_i]
                    count_ef[(e_i, f_i)] += c
                    total_f[f_i] += c
                    count_a[(i, j, l_e, l_f)] += c
                    total_a[(j, l_e, l_f)] += c

            if n %1000 == 0:
                sys.stderr.write('\t\t\tAt iteration %i of %i ...\n' % (n, len(bitext)))

        sys.stderr.write('\t\tEstimating Proabilities...\n')
        # estimate probabilities
        # for (n, (f, e)) in enumerate(bitext):
        #
        #     l_e = len(e)
        #     l_f = len(f)
        #     for (j, e_i) in enumerate(e, 1):
        #         for (i, f_i) in enumerate(f, 1):
        #             t[(e_i, f_i)] = 0
        #             a[(i, j, l_e, l_f)] = 0

        for (n, (f, e)) in enumerate(bitext):
            l_e = len(e)
            l_f = len(f)
            for (j, e_i) in enumerate(e, 1):
                for (i, f_i) in enumerate(f, 1):
                    t[(e_i, f_i)] = count_ef[(e_i, f_i)] / total_f[f_i]
                    a[(i, j, l_e, l_f)] = Decimal(count_a[(i, j, l_e, l_f)]) / \
                                              Decimal(total_a[(j, l_e, l_f)])
            if n %1000 == 0:
                sys.stderr.write('\t\t\tAt iteration %i of %i ...\n' % (n, len(bitext)))

    return (t, a)


def align(t, a, bitext):
    # alignment
    sys.stderr.write('Beginning alignment...\n')
    for (f, e) in bitext:
        l_e = len(e)
        l_f = len(f)
        for (j, e_i) in enumerate(e):
            max_value = -1
            max_align = 0
            for (i, f_i) in enumerate(f):
                # choose highest probability of all alignments
                value = t[(e_i, f_i)] * a[(i, j, l_e, l_f)]
                if max_value < value:
                    max_value = value
                    max_align = i
            sys.stdout.write("%i-%i " % (max_align, j))
        sys.stdout.write("\n")

if __name__ == '__main__':
    b = m1.add_null(bitext)
    (t, a) = train_model(b, 1)
    align(t,a,b)
