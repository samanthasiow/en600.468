#!/usr/bin/env python
import optparse
import sys

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
for s in xrange(0, num_sents):
  hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
  (best_score, best) = (-1e300, '')
  for (num, hyp, feats) in hyps_for_one_sent:
    score = 0.0
    for feat in feats.split(' '):
      (k, v) = feat.split('=')
      print 'feature key', k, 'feature value', v
      score += weights[k] * float(v)
    if score > best_score:
      (best_score, best) = (score, hyp)
  try:
    sys.stdout.write("%s\n" % best)
  except (Exception):
    sys.exit(1)
