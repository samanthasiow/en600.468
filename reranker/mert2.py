#!/usr/bin/env python
import optparse
import sys
import random
import bleu

optparser = optparse.OptionParser()
optparser.add_option("-k", "--kbest-list", dest="input", default="data/dev+test.100best", help="100-best translation lists")
optparser.add_option("-l", "--lm", dest="lm", default=-1.0, type="float", help="Language model weight")
optparser.add_option("-r", "--ref-list", dest="ref", default="data/dev+train.ref", help="Reference translation lists")
optparser.add_option("-t", "--tm1", dest="tm1", default=-0.5, type="float", help="Translation model p(e|f) weight")
optparser.add_option("-s", "--tm2", dest="tm2", default=-0.5, type="float", help="Lexical translation model p_lex(f|e) weight")
(opts, _) = optparser.parse_args()
weights = {'p(e)'       : float(opts.lm) ,
           'p(e|f)'     : float(opts.tm1),
           'p_lex(f|e)' : float(opts.tm2)}

all_hyps = [pair.split(' ||| ') for pair in open(opts.input)]
all_refs = [ref.strip() for ref in open(opts.ref)]
num_sents = len(all_hyps) / 100

# return slope of line for each translation
def compute_line(translation):
    return


def compute_score(weights, refs, hyps):
    tot_stats = [0 for i in xrange(10)]
    hyp_list = []
    for s in xrange(0, num_sents):
        hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
        (best_score, best) = (-1e300, '')
        for (num, hyp, feats) in hyps_for_one_sent:
            score = 0.0
            for feat in feats.split(' '):
                (k, v) = feat.split('=')
                score += weights[k] * float(v)
            if score > best_score:
                (best_score, best) = (score, hyp)
        hyp_list.append("%s\n" % best)
    for (r,h) in zip(refs, hyp_list):
        tot_stats = [sum(s) for s in zip(tot_stats, bleu.bleu_stats(r, h))]
        # for i in xrange(len(tot_stats)):
        #     tot_stats[i] += int(best[i])
    return bleu.bleu(tot_stats)

def get_interval(l):
    point1 = l.pop()
    point2 = l.pop()
    return (point1, point2)

def minimum_error_rate_training(weights, all_hyps, num_sents):
    # # repeat till convergence
    # # for all parameters
    # weight_hypothesis = [weights.copy()] #inialize the possible weights
    rand_weights = {
        'p(e)'       : random.uniform(-3,3),
        'p(e|f)'     : random.uniform(-3,3),
        'p_lex(f|e)' : random.uniform(-3,3)}
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
    curr_score = compute_score(rand_weights,  all_refs, all_hyps)
    for w in rand_weights:

        print weights
        # set of threshold points T
        threshold_set = []
        # for all sentences
        for s in xrange(0, num_sents):
            print 'for sentence', s
            reference = all_refs[s]
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
                        alt_weight_sum += float(rand_weights[k])
                y_intersect = float(alt_weight_sum * gradient)
                print 'gradient', gradient, 'combined weight', alt_weight_sum
                # line = (gradient, y_intersect,
                #           hypothesis, sentence number for reference)
                line = {'m': gradient, 'c': y_intersect, 'hyp': hyp, 'ref': reference}
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
            # while find line l_2 that intersects with l first
            while i+1 < len(sorted_hyp_lines):
                # intersection points in order
                intersection_points = {}
                l_1 = sorted_hyp_lines[i] # y = ax + c

                # find line l_2 that intersects with l_1 first
                for j in xrange(i+1, len(sorted_hyp_lines)):
                    l_2 = sorted_hyp_lines[j] # y = bx + d
                    # Check if m is the same (lines are parallel, take the line with higher c)
                    if l_1['m'] == l_2['m']:
                        continue
                    # intersection point x,y
                    # x = (d-c)/(a-b)
                    x_numerator = float(l_2['c']) - float(l_1['c'])
                    x_denominator = float(l_1['m']) - float(l_2['m'])
                    x = float(x_numerator / x_denominator)
                    # y = a(x) + c
                    y = l_1['m'] * x + l_1['c']
                    # save all intersection points of other lines with l_1
                    # [0]: x, [1]: y, [2]: (l_1['hyp'], l_2['hyp']), [3]: reference
                    intersection_points[(x,y,(l_1['hyp'], l_2['hyp']),reference)] = j

                if len(intersection_points) == 0:
                    print 'finished calculating upper envelope'
                    break
                else:
                    # minimum intersection point with l_1 = first intersection with l_2
                    min_line_intersect = min(intersection_points)
                    upper_envelope.append(min_line_intersect)

                    # l = l_2
                    i = intersection_points[min_line_intersect]
            # add parameter value at intersection
            # parameter points in the format:
            # x_1, x_2, bleu score, tuple(hypothesis 1, ref)
            # where x_1 is the start of the interval, and x_2 is the end of the interval
            for index, point in enumerate(upper_envelope):
                # first point starts at infinity
                if index == 0:
                    parameter = {
                                    'x_1': float('-inf'),
                                    'x_2': point[1],
                                     'score': bleu.bleu_stats(point[2][0], point[3]),
                                     'hyp': point[2][0],
                                     'ref': point[3]
                                 }
                else:
                    parameter = {
                                    'x_1': previous_x,
                                    'x_2': point[1],
                                    'score': bleu.bleu_stats(point[2][0], point[3]),
                                    'hyp': point[2][0],
                                    'ref': point[3]
                                 }
                threshold_set.append(parameter)
                previous_x = point[1]
                # last point ends at infinity
                if index+1 == len(upper_envelope):
                    parameter = {
                                    'x_1': previous_x,
                                    'x_2': float('-inf'),
                                    'score': bleu.bleu_stats(point[2][1], point[3]),
                                    'hyp': point[2][1],
                                    'ref': point[3]
                                 }
                    threshold_set.append(parameter)


        # --- END SAMANTHA'S STUFF ---
        # sort T by parameter value
        # compute score for value before first threshold point
        # for all t in T
            # compute score for value after t
            # if score is highest
                # record max score and t
        # if max score > current score
            # update parameter value

        #sort threshold set based on the bleu score
        threshold_set = sorted(threshold_set, key=lambda x: x["score"])
        points = [(x1, x2) for x1, x2, score, hyp, ref in threshold_set]
        point_list = []
        for point in points:
            point_list.append(point[0])
            point_list.append(point[1])

        point_list.sort()

        t_weights = rand_weights
        start, end = get_interval(point_list)

        max_score = compute_score(rand_weights)
        while point_list:
            val = (start+end)/float(2)
            t_weights[w] = val
            score = compute_score(t_weights, all_refs, all_hyps)
            if score > max_score:
                max_score = score
                best_val = val
            start = end
            end = point_list.pop()

        if max_score > curr_score:
            curr_score = max_score
            best_w = w
            best_v = best_val
            weights[w] = best_v


    for s in xrange(0, num_sents):
      hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
      (best_score, best) = (-1e300, '')
      for (num, hyp, feats) in hyps_for_one_sent:
        score = 0.0
        for feat in feats.split(' '):
          (k, v) = feat.split('=')
          score += weights[k] * float(v)
        if score > best_score:
          (best_score, best) = (score, hyp)
      try:
        sys.stdout.write("%s\n" % best)
      except (Exception):
        sys.exit(1)



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
