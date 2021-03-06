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

def swap(sentence, sentence_indeces, pos1, pos2):
  new_sentence = ()
  for i in range(len(sentence)):
    if i == pos2:
      new_sentence += (sentence[pos2],)
      new_sentence += (sentence[pos1],)
      swp_index = sentence_indeces[pos2]
      sentence_indeces[pos2] = sentence_indeces[pos1]
      sentence_indeces[pos1] = swp_index
    elif i != pos1:
      new_sentence += (sentence[i],)
  return new_sentence, sentence_indeces


tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0, word)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of
  # the first i words of the input sentence. You should generalize
  # this so that they can represent translations of *any* i words.

  # create array. each index corresponding to a boolean of whether the foreign
  # word at that index has been translated in the original sentence
  translated = []
  for x in range(len(f)):
    translated.append(False)

  hypothesis = namedtuple("hypothesis", "logprob, lm_state, phrase, predecessor")
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None)
  stacks = [{} for _ in f] + [{}]
  stacks[0][lm.begin()] = initial_hypothesis
  # for each position in sentence f
  for i, stack in enumerate(stacks[:-1]):

    # generate the next sentence. translated words in the beginning and non
    # translated words at the end
    new_sentence = ()
    new_sentence_indeces = []
    count = 0
    for index in range(len(f)):
      if translated[index] is True:
        new_sentence += (f[index],)
        new_sentence_indeces.append(index)
    for index in range(len(f)):
      if translated[index] is False:
        new_sentence += (f[index],)
        new_sentence_indeces.append(index)
        count +=1

    # for each hypothesis in the stack
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune
      # dictionary mapping hypotheses to list of the indeces of the words in
      # the original foreign sentence
      h_dict = {}
      # for all translation options (no swap)
      for j in xrange(i+1,len(f)+1):
        if new_sentence[i:j] in tm:
          for phrase in tm[new_sentence[i:j]]:
            logprob = h.logprob + phrase.logprob
            lm_state = h.lm_state
            for word in phrase.english.split():
              (lm_state, word_logprob) = lm.score(lm_state, word)
              logprob += word_logprob
            logprob += lm.end(lm_state) if j == len(new_sentence) else 0.0
            new_hypothesis = hypothesis(logprob, lm_state, phrase, h)
            if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob: # second case is recombination
              # add the index of the word in the original sentence to a list
              phrase_indeces = []
              for index in xrange(i,j):
                phrase_indeces.append(new_sentence_indeces[index])
              h_dict[new_hypothesis] = phrase_indeces
      # generate new sentence and indeces for swapped words
      new_sentence, new_sentence_indeces = swap(new_sentence, new_sentence_indeces, i, i+1)
      for j in xrange(i+1,len(f)+1):
        if new_sentence[i:j] in tm:
          for phrase in tm[new_sentence[i:j]]:
            logprob = h.logprob + phrase.logprob
            lm_state = h.lm_state
            for word in phrase.english.split():
              (lm_state, word_logprob) = lm.score(lm_state, word)
              logprob += word_logprob
            logprob += lm.end(lm_state) if j == len(new_sentence) else 0.0
            new_hypothesis = hypothesis(logprob, lm_state, phrase, h)
            if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob: # second case is recombination
              # add the index of the word in the original sentence to a list
              phrase_indeces = []
              for index in xrange(i,j):
                phrase_indeces.append(new_sentence_indeces[index])
              h_dict[new_hypothesis] = phrase_indeces
      # get the best hypothesis
      best = sorted(h_dict.iterkeys(),key=lambda h: -h.logprob)[:1]
      # get the indeces in the original sentence
      phrase = h_dict[best[0]]
      # set the words as translated
      for index in phrase:
        translated[index] = True
      # place in the correct slot in the stack
      count = 0
      for word in translated:
        if word is True:
          count += 1
      stacks[count][best[0].lm_state] = best[0]
  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)


  # returns the highest likelihood translated phrase
  def extract_english(h):
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  print extract_english(winner)

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)

    # test with different order

    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))

def get_neighbor_hypotheses(h):
    return
