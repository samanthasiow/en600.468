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
        print weights
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
                gradient = 0.0 # gradient = value of the feature
                y_intersect = 0.0 # y_intersect = sum of other weights * value of feature
                alt_weight_sum = 0.0
                for feat in feats.split(' '):
                    (k, v) = feat.split('=')
                    print '\t\tfeature key', k, 'feature value', v
                    # get the parameter that we are interested in
                    if k == w:
                        gradient = float(v)
                    else:
                        alt_weight_sum += float(weights[k])
                y_intersect = float(alt_weight_sum * gradient)
                print 'gradient', gradient, 'combined weight', alt_weight_sum
                # line = (gradient, y_intersect,
                #           hypothesis, sentence number for reference)
                line = {'m': gradient, 'c': y_intersect, 'hyp': hyp, 'line_num': s}
                hyp_lines.append(line)
                # sort lines in descending order,
                # with steepest gradient first, then sort by y intersection
            sorted_hyp_lines = sorted(hyp_lines, key=lambda element: (-element['m'], -element['c']))
            # get steepest lines
            steepest_lines = {}
            for i,line in enumerate(sorted_hyp_lines):
                if line['m'] in steepest_lines:
                    if line['c'] > steepest_lines[line['m']]['c']:
                        steepest_lines[line['m']] = line
                else:
                    steepest_lines[line['m']] = line
            # find upper envelope:
            upper_envelope = []
            i = 0
            while i+1 < len(sorted_hyp_lines):
            # while find line l_2 that intersects with l first
                # TODO: Check if m is the same (lines are parallel, take the line with higher c)
                l_1 = sorted_hyp_lines[i] # y = ax + c
                l_2 = sorted_hyp_lines[i+1] # y = bx + d
                if l_1['m'] == l_2['m']:
                    i += 1
                    continue
                # intersection point x,y
                # x = (d-c)/(a-b)
                x_numerator = l_2['c'] - l_1['c']
                x_denominator = l_1['m'] - l_2['m']
                x = float(x_numerator / x_denominator)
                # y = a(x) + c
                y = l_1['m'] * x + l_1['c']

                i += 1

                # add parameter value at intersection to T
                # l = l_2
        # --- END SAMANTHA'S STUFF ---
        # sort T by parameter value
        # compute score for value before first threshold point
        # for all t in T
            # compute score for value after t
            # if score is highest
                # record max score and t
        # if max score > current score
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
