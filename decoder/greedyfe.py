#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)

lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]
MAXIMUM_MOVE_DISTANCE = 3

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0, word)]

def extract_f_phrases(h):
    phrases = []
    while h.predecessor is not None:
        split = h.phrase.french.split()
        phrases.append(tuple(split))
        h = h.predecessor
    # reverse list
    return tuple(phrases[::-1])

def extract_e_phrases(h):
    phrases = []
    while h.predecessor is not None:
        phrases.append(h.phrase.english)
        h = h.predecessor
    # reverse list
    return phrases[::-1]

# The following code implements a monotone decoding
# algorithm (one that doesn't permute the target phrases).
# Hence all hypotheses in stacks[i] represent translations of
# the first i words of the input sentence. You should generalize
# this so that they can represent translations of *any* i words.
def decode_f_to_e(f):
    hypothesis = namedtuple("hypothesis", "logprob, lm_state, phrase, predecessor")
    initial_hypothesis = hypothesis(0.0, lm.begin(), None, None)
    stacks = [{} for _ in f] + [{}]
    stacks[0][lm.begin()] = initial_hypothesis
    for i, stack in enumerate(stacks[:-1]):
        for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
            for j in xrange(i+1,len(f)+1):
                if f[i:j] in tm:
                    for phrase in tm[f[i:j]]:
                        logprob = h.logprob + phrase.logprob
                        lm_state = h.lm_state
                    for word in phrase.english.split():
                        (lm_state, word_logprob) = lm.score(lm_state, word)
                        logprob += word_logprob
                    logprob += lm.end(lm_state) if j == len(f) else 0.0
                    new_hypothesis = hypothesis(logprob, lm_state, phrase, h)
                    if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob: # second case is recombination
                        stacks[j][lm_state] = new_hypothesis
    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
    return winner

def translate_f_hypothesis(f):
    e_hypothesis = ''
    for phrases in f:
        if tuple(phrases) in tm:
            max_logprob = -1000
            max_e = ''
            for phrase in tm[tuple(phrases)]:
                if phrase.logprob > max_logprob:
                    max_e = phrase.english
                    max_logprob = phrase.logprob
            e_hypothesis += max_e + ' '
        else:
            if len(phrases) > 1:
                for p in phrases:
                    e_hypothesis += p + ' '
            elif len(phrases) == 1:
                e_hypothesis += phrases + ' '
            else:
                continue
    return e_hypothesis[:-1]

def get_neighbor_hypotheses(h):
    f_phrases = extract_f_phrases(h)
    e_phrases = extract_e_phrases(h)
    neighbor_f_hypotheses = set()
    neighbor_e_hypotheses = set()

    # operations:
    # move, swap, replace, split, merge, (bi-replace)
    for i, e in enumerate(e_phrases):
        # move: move each phrases of output within some set distance from origin
        # move 3 forward
        if i+MAXIMUM_MOVE_DISTANCE < len(e_phrases):
            new_h = []
            for pre in e_phrases[:i]:
                new_h.append(pre)
            for between in e_phrases[i+1:i+1+MAXIMUM_MOVE_DISTANCE]:
                new_h.append(between)
            new_h.append(e)
            for post in e_phrases[i+1+MAXIMUM_MOVE_DISTANCE:]:
                new_h.append(post)
            neighbor_e_hypotheses.add(tuple(new_h))
        # move 3 backward
        if i-MAXIMUM_MOVE_DISTANCE > 0:
            new_h = []
            for pre in e_phrases[:i-MAXIMUM_MOVE_DISTANCE]:
                new_h.append(pre)
            new_h.append(e)
            for between in e_phrases[i-MAXIMUM_MOVE_DISTANCE:i]:
                new_h.append(between)
            for post in e_phrases[i+1:]:
                new_h.append(post)
            neighbor_e_hypotheses.add(tuple(new_h))

        # swap: swap two adjacent phrases of output with each other
        if i+1 < len(e_phrases):
            new_h = []
            for pre in e_phrases[:i]:
                new_h.append(pre)
            new_h.append(e_phrases[i+1])
            new_h.append(e_phrases[i])
            for post in e_phrases[i+2:]:
                new_h.append(post)
            neighbor_e_hypotheses.add(tuple(new_h))

    for i,f in enumerate(f_phrases):
        # split: split source phrase into two parts, and translate separately
        if len(f) > 1:
            new_h = []
            split_index = len(f) / 2
            for pre in f_phrases[:i]:
                new_h.append(pre)
            new_h.append(f[:split_index+1])
            new_h.append(f[split_index+1:])
            for post in f_phrases[i+1:]:
                new_h.append(post)
            neighbor_f_hypotheses.add(tuple(new_h))

        # merge: merge two adjacent source phrases into one, and translate as a whole
        if i+1 < len(f_phrases):
            new_h = []
            for pre in f_phrases[:i]:
                new_h.append(pre)
            new_h.append(tuple(f_phrases[i]) + tuple(f_phrases[i+1]))
            for post in f_phrases[i+2:]:
                new_h.append(post)
            neighbor_f_hypotheses.add(tuple(new_h))

    # replace: replace translated phrase with another phrase
    for w in f_phrases:
        new_h = ''
        for phrase in f_phrases:
            if tuple(phrase) in tm:
                if len(tm[tuple(phrase)]) > 1 and w == phrase:
                    print 'found second best option', tm[tuple(phrase)][1].english
                    new_h += tm[tuple(phrase)][1].english + ' '
                else:
                    new_h += tm[tuple(phrase)][0].english + ' '
            else:
                if len(phrase) > 1:
                    for p in phrase:
                        new_h += p + ' '
                elif len(phrase) == 1:
                    new_h += phrase + ' '
                else:
                    continue
        neighbor_e_hypotheses.add(new_h[:-1])

    # bi-replace: replace translated phrase with another phrase
    for i,w in enumerate(f_phrases):
        new_h = ''
        for j,phrase in enumerate(f_phrases):
            if tuple(phrase) in tm:
                if len(tm[tuple(phrase)]) > 1 and i == j:
                    new_h += tm[tuple(phrase)][1].english + ' '
                elif j == i + 1 and len(tm[tuple(f_phrases[j])]) > 1:
                    new_h += tm[tuple(phrase)][1].english + ' '
                else:
                    new_h += tm[tuple(phrase)][0].english + ' '
            else:
                if len(phrase) > 1:
                    for p in phrase:
                        new_h += p + ' '
                elif len(phrase) == 1:
                    new_h += phrase + ' '
                else:
                    continue
        neighbor_e_hypotheses.add(new_h[:-1])

    for f in neighbor_f_hypotheses:
        neighbor_e_hypotheses.add(translate_f_hypothesis(f))

    return neighbor_e_hypotheses


if __name__ == "__main__":
    sys.stderr.write("Decoding %s...\n" % (opts.input,))

    for f in french:
        winner = decode_f_to_e(f)

        # returns the highest likelihood translated phrase
        def extract_english(h):
            if h.predecessor is None:
                return ""
            else:
                return "%s%s " % (extract_english(h.predecessor), h.phrase.english)
        winner_phrase = extract_english(winner)
        e_neighbors = get_neighbor_hypotheses(winner)

        max_logprob = winner.logprob
        max_neighbor = winner_phrase
        lm_state = lm.begin() # initial state is always <s>
        for sentence in e_neighbors:
            hyp = ''
            for word in sentence:
                hyp += word + ' '
            logprob = 0.0
            for word in hyp.split():
              (lm_state, word_logprob) = lm.score(lm_state, word)
              logprob += word_logprob
            logprob += lm.end(lm_state) # transition to </s>, can also use lm.score(lm_state, "</s>")[1]
            if logprob > max_logprob:
                max_logprob = logprob
                max_neighbor = hyp

        print max_neighbor

      # add phrase to an array that represents all phrases in the translated sentence.
      # while op score > current score:
      #     for op in operation:
      #         get score of operation on all phrases
      #     implement highest scoring op
        if opts.verbose:
            def extract_tm_logprob(h):
                return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
            tm_logprob = extract_tm_logprob(winner)
            sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
              (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
