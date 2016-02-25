#!/usr/bin/env python
''' IBM Model 1 with Combined Predictions for E->F and F->E models. '''
from __future__ import division
import optparse
import sys
import collections
import decimal
import model1 as IBMModel1
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

''' Reversing bitext to calculate reversed order model.'''
def reverse_bitext(b):
    sys.stderr.write('Reversing bitext...\n')
    reversed_bitext = []
    for (f,e) in b:
        reversed_bitext.append([e,f])

    return reversed_bitext

''' Creating combined model from bitext. '''
def combined_train(bitext_raw):
    reversed_bitext_raw = reverse_bitext(bitext_raw)

    bitext = IBMModel1.add_null(bitext_raw)
    reversed_bitext = IBMModel1.add_null(reversed_bitext_raw)

    # t_b: translation probability of e->f text
    sys.stderr.write('Training forward model...\n')
    t_b = IBMModel1.train_model(bitext, opts.iterations)
    # t_r: translation probability of f->e text
    sys.stderr.write('Training reverse model...\n')
    t_r = IBMModel1.train_model(reversed_bitext, opts.iterations)

    combined_t = defaultdict(Decimal)
    sys.stderr.write('Calculating combined translation probability...\n')
    for p_ef in t_b:
        (e,f) = p_ef
        combined_t[(e,f)] = t_b[(e,f)] * t_r[(f,e)]

    return combined_t

if __name__ == '__main__':
    # Contains an array of all sentences in the text
    # Each element is a 2-element array, [0] is the french translation of the sentence, and [1] is the english translation.
    bitext_raw = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
    bitext = IBMModel1.add_null(bitext_raw)

    combined_t = combined_train(bitext_raw)
    IBMModel1.align(combined_t, bitext)
