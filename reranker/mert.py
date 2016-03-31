#!/usr/bin/env python
import optparse
import sys
import random

optparser = optparse.OptionParser()
optparser.add_option("-k", "--kbest-list", dest="input", default="data/dev+test.100best", help="100-best translation lists")
optparser.add_option("-l", "--lm", dest="lm", default=-1.0, type="float", help="Language model weight")
optparser.add_option("-t", "--tm1", dest="tm1", default=-0.5, type="float", help="Translation model p(e|f) weight")
optparser.add_option("-s", "--tm2", dest="tm2", default=-0.5, type="float", help="Lexical translation model p_lex(f|e) weight")
(opts, _) = optparser.parse_args()
weights = {'p(e)'       : float(opts.lm) ,
           'p(e|f)'     : float(opts.tm1),
           'p_lex(f|e)' : float(opts.tm2)}

all_hyps = [pair.split(' ||| ') for pair in open(opts.input)]
num_sents = len(all_hyps) / 100

# return slope of line for each translation
def compute_line(translation):
    return


def minimum_error_rate_training(weights, all_hyps, num_sents):
    # # repeat till convergence
    # # for all parameters
    # weight_hypothesis = [weights.copy()] #inialize the possible weights
    # rand_weights = {
    #     'p(e)'       : random.uniform(-3,3),
    #     'p(e|f)'     : random.uniform(-3,3),
    #     'p_lex(f|e)' : random.uniform(-3,3)}
    # # append randomized weights to hypothesis
    # weight_hypothesis.append(rand_weights)
    # # for each weight hypothesis
    # for w_hyp in weight_hypothesis:
    #     # set of threshold points T
    #     threshold_set = set()
    #     # for all sentences
    #     for s in xrange(0, num_sents):
    #         hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
    #         # for all translations
    #         for t in hyps_for_one_sent:
    #             compute_line(t)

    # for all parameters
    for w in weights:
        # set of threshold points T
        threshold_set = set()
        # for all sentences
        for s in xrange(0, num_sents):
            print 'for sentence', s
            # for all translations
            hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
            hyp_lines = []
            for (num, hyp, feats) in hyps_for_one_sent:
                print '\tnum', num, 'hyp', hyp
                # get slope and intersection to define line
                slope = 0.0 # slope = value of the feature
                intersection = 0.0 # intersection = sum of other weights * value of feature
                alt_weight_sum = 0.0
                for feat in feats.split(' '):
                  (k, v) = feat.split('=')
                  print '\t\tfeature key', k, 'feature value', v
                  # get the parameter that we are interested in
                  if k == w:
                      slope = float(v)
                  else:
                      alt_weight_sum += float(weights[k])
                intersection = float(alt_weight_sum * slope)
                



                # compute line l: parameter value --> score
            # find l with steepest descent
            # find upper envelope:
            # while find line l_2 that intersects with l first
                # add parameter value at intersection to T
                # l = l_2
        # sort T by parameter value
        # compute score for value before first threshold point
        # for all t in T
            # compute score for value after t
            # if score is highest
                # record max score and t
        # if max scaore > current score
            # update parameter value

# if "__name__" == "__main__":
minimum_error_rate_training(weights, all_hyps, num_sents)

# for s in xrange(0, num_sents):
#     for i in xrange(0, 100): # iterate n times
#         # randomize starting feature weights
#         rand_weights = {
#             'p(e)'       : random.uniform(-3,3),
#             'p(e|f)'     : random.uniform(-3,3),
#             'p_lex(f|e)' : random.uniform(-3,3)}
#         # iterate till convergence
#
#   hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
#   (best_score, best) = (-1e300, '')
#   for (num, hyp, feats) in hyps_for_one_sent:
#     score = 0.0
#     for feat in feats.split(' '):
#       (k, v) = feat.split('=')
#       print 'feature key', k, 'feature value', v
#       score += weights[k] * float(v)
#     if score > best_score:
#       (best_score, best) = (score, hyp)
#   try:
#     sys.stdout.write("%s\n" % best)
#   except (Exception):
#     sys.exit(1)
