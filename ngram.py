import utils
import numpy as np
import pandas as pd
from collections import Counter
from time import time 
import random
import utils
import numpy as np
import pandas as pd
from collections import Counter

import utils
import numpy as np
import pandas as pd
from collections import Counter
from time import time 

import utils
import numpy as np
import pandas as pd
from collections import Counter

class Ngram():
    def __init__(self, n, version, verbose=True):
        self.n = n 
        self.version_control(version)
        self.version = version
        self.verboseprint = print if verbose else lambda *a, **k: None
        self.verboseprint('Running '+version+' {}-gram model...'.format(n))
        
    def version_control(self, version):
        if version not in ['classical', 'turing', 'backoff'] and 'add' not in version and 'kns' not in version:
            raise ValueError("Please choose between 'add-k', 'kns-k', 'backoff', classical' or 'turing'")
        if 'add' in version:
            self.k = int(version.split('-')[1])
        elif 'kns' in version:
            self.d = int(version.split('-')[1])
        
    def ngram(self, s):
        """
        For a given sentence s, return list of n-grams
        """
        # Break sentence in the token, remove empty tokens
        tokens = [token for token in s.split(" ") if token != ""]

        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
        ngrams = zip(*[tokens[i:] for i in range(self.n)])
        return [" ".join(ngram) for ngram in ngrams]

    def generate_ngrams(self, data):
        """
        For a given data set of sentences s, return list of lists of n-grams
        """
        ngrams = []
        for sentence in data.text:
            ngrams.append(self.ngram(sentence))
        return ngrams
    
    def compute_probabilities(self, data):
        """
        For a given data set of sentences s, compute the probabilities used in the n-gram
        model with the occurences. 
        """
        self.t0 = time()
        self.ngrams = utils.flatten(self.generate_ngrams(data))
        # Contexts are w1...wt-1 for a sentence w1...wt and wt the last word of a given n-gram
        self.contexts = [utils.split_and_join(self.ngrams[i], self.n-1) for i in range(len(self.ngrams))]
        # Size of corpus
        words = ' '.join(self.ngrams).split()
        self.V = len(set(words)) - 2 # remove characters <s> and </s>
        self.unigram_count = dict(Counter(words))
        # Occurences of each n-gram 
        self.ngram_count = dict(Counter(self.ngrams))
        # Occurences of each contexts 
        self.context_count = dict(Counter(self.contexts))
        # Number of unique n-grams
        N_gram = len(set(self.ngram_count.keys()))
        # Number of unique contexts
        N_context = len(set(self.context_count.keys()))
        self.C_mul = N_context / N_gram
        
        # Probabilities are dictionnaries of type (ngram: proba)
        ## Laplace smoothing (lecture notes)
        if 'add' in self.version:
            self.probabilities = {
                ngram: (self.ngram_count[ngram]+self.k) / (self.context_count[context]+self.k*self.V)
                for ngram, context in zip(self.ngrams, self.contexts)
            }
            
        ## Classical n-gram (without smoothing)
        elif self.version == 'classical':
            self.probabilities = {
                ngram: self.C_mul * self.ngram_count[ngram] / self.context_count[context] 
                for ngram, context in zip(self.ngrams, self.contexts)
            }
            
        ## Good-Turing estimation (lectures notes)
        elif self.version == 'turing':
            self.turing_gram_freq = self.good_turing(self.ngram_count)
            N = sum(self.turing_gram_freq.values())
            self.probabilities = {
                ngram: self.new_turing_count(ngram, context, N)
                for ngram, context in zip(self.ngrams, self.contexts)
            }
            
        
        ## Kneser-Ney Smoothing (lectures notes)
        elif 'kns' in self.version:
            # First, we need to store the contexts counts for each t. Namely, for t=n, we store the n-gram counts.
            # Then, for t=n-1, we store (in self.CONTEXTS_COUNT[1]=self.CONTEXTS_COUNT[self.n-t]) the contexts counts
            # composed of the first (n-1) grams. Finally, for t=1, we store the unigram counts. 
            # In KNS algorithm, lambda is (up to the discount factor d) the division of two guys:
            # - denominator: the total number of occurences of the context w(t-n+1)...w(t-1)
            # - numerator: the total number of words that can appear after w(t-n+1)...w(t-1)
            # One could think that those quantities are equal, but they aren't. Indeed, there often exists
            # the same word w such that w(t-n+1)...w(t-1)w exists. In that case, there are redundant contexts
            # w(t-n+1)...w(t-1) in a sense that there are the contexts of the same last word. The numerator in 
            # lambda is in fact the number of unique words that can appear after w(t-n+1)...w(t-1). So, we
            # need to make a distinction between the countings. That's what we do in introducing the variables
            # self.UNIQUE_CONTEXTS and self.CONTEXTS_UNIQUE_COUNT. The only difference in the code is that 
            # unique contexts are built using a loop over set(self.ngrams) and not self.ngrams, which is now
            # a set of unique n-grams. Thus, there are no redundant n-grams and so we can compute the count
            # of unique contexts in the sense that: for any context c, there exists only one token w such that
            # cw is unique. 
            self.CONTEXTS = []; self.UNIQUE_CONTEXTS = []; self.CONTEXTS_COUNT = []; self.CONTEXTS_UNIQUE_COUNT = []
            for t in range(self.n, 0, -1):
                # Occurences of each contexts from t=n to t=1
                self.CONTEXTS.append([utils.split_and_join(ngram, t) for ngram in self.ngrams])
                self.CONTEXTS_COUNT.append(dict(Counter(self.CONTEXTS[self.n-t])))
                # Occurences of each "unique" contexts from t=n to t=1
                self.UNIQUE_CONTEXTS.append([utils.split_and_join(ngram, t) for ngram in set(self.ngrams)])
                self.CONTEXTS_UNIQUE_COUNT.append(dict(Counter(self.UNIQUE_CONTEXTS[self.n-t])))
            self.probabilities = {}
            for ngram in set(self.ngrams):
                self.probabilities[ngram] = self.p_KSN(ngram, self.n, self.d)
            
        ## Stupid Backoff
        elif self.version == 'backoff':
            self.CONTEXTS = []; self.CONTEXTS_COUNT = []
            self.ngrams = utils.flatten(self.generate_ngrams(data))
            for t in range(self.n, 0, -1):
                # Occurences of each contexts from t=n to t=1
                self.CONTEXTS.append([utils.split_and_join(ngram, t) for ngram in self.ngrams])
                self.CONTEXTS_COUNT.append(dict(Counter(self.CONTEXTS[self.n-t])))
            self.probabilities = {}
                    
    def p_BO(self, w_t, t):
        if t==1 and w_t in self.unigram_count:
            return self.unigram_count[w_t]/self.V
        elif t==1 and w_t not in self.unigram_count:
            return 1/self.V
        else:
            if w_t in self.CONTEXTS_COUNT[self.n-t].keys():
                w_t_1 = w_t.split()[:-1] 
                w_t_1 = " ".join(w_t_1)
                return self.CONTEXTS_COUNT[self.n-t][w_t] / self.CONTEXTS_COUNT[self.n-t+1][w_t_1]
            else:
                w_t_1 = w_t.split()[1:] 
                w_t_1 = " ".join(w_t_1)
                return 0.4 * self.p_BO(w_t_1, t-1)
            

    # Specific methods for each version of N-gram models
    ## Good-Turing
    def good_turing(self, dict_count):
        """
        For a given dictionary of type {ngram: count}, returns the dictionary {c: Nc} following
        the lectures notations. Nc is the number of n-grams that appear exactly c times ("counts of counts")
        """
        # First, build the dictionary of frequencies {c: Nc}
        freq_ = {i: len([v for v in dict_count.values() if v == i]) for i in set(dict_count.values())}
        # Then, make the exclusion to only keep keys (i.e. number of times c) that are not in the previous 
        # dictionary (see in the lecture: what about if Nc>0 and Nc+1=0
        exclusion = list(set(range(1, max(freq_.keys())+2)) - set(dict_count.values()))
        # Now he goal here is to add the items {Nc+1: 0} as those frequencies are such that none of ngrams
        # can be found exactly Nc+1 times
        freq = freq_.copy()
        for i in exclusion:
            freq[i] = 0
        # Compute the ratios Nc+1 / Nc
        new_freq = {i: freq[i+1]/freq[i] for i in freq_.keys()}
        return new_freq
        
    def new_turing_count(self, ngram, context, N):
        """
        Implements the new counting: c* = (c+1) * Nc+1 / Nc if c < 30 else the classical counting
        :param N: int, total sum of counts of counts
        """
        c = self.ngram_count[ngram]
        if c < 30:
            return (c+1) * self.turing_gram_freq[c] / N  
        else:
            return self.C_mul * self.ngram_count[ngram] / self.context_count[context] 
        
                
    ## Kneser-Ney
    def f_t(self, w_t, w_t_1, t, d):
        """
        Compute f_t (see lecture notes)
        """
        return max(self.CONTEXTS_COUNT[self.n-t][w_t]-d, 0) / self.CONTEXTS_COUNT[self.n-t+1][w_t_1]

    def l_t(self, w_t_1, t, d):
        """
        Compute lambda_t (see lecture notes). We make use of self.CONTEXTS_UNIQUE_COUNT.
        """
        card = self.CONTEXTS_UNIQUE_COUNT[self.n-t+1][w_t_1]
        return d * card / self.CONTEXTS_COUNT[self.n-t+1][w_t_1]

    def p_KSN(self, w, t, d):
        """
        Recursively compute P_t (see lecture notes).
        """
        if t == 1:
            return 1
        else:
            # Gram of interest
            w_t = utils.split_and_join(w, t)
            # Context associated to w_t
            w_t_1 = utils.split_and_join(w, t-1)
            return self.f_t(w_t, w_t_1, t, d) + self.l_t(w_t_1, t, d) * self.p_KSN(w_t_1, t-1, d)

    def compute_sentence_probability(self, sentence):
        """
        For a given sentence, compute its probability using the chain rule and exp-log trick
        """
        ngrams = self.ngram(sentence)
        p = 0
        for ngram in ngrams:
            if self.version == 'backoff':
                proba = self.p_BO(ngram, self.n)
                self.probabilities[ngram] = proba
                p += np.log(proba)
            else:
                # except is used in case where some n-grams cannot be found in the test set
                try:
                    p += np.log(self.probabilities[ngram])
                except:
                    # check if a N-gram is not equal to [wt wt-1 ... <s> <\s> wt-k...w1] (crossing sentences)
                    if all(k in ngram for k in ['<s>', '<\s>']):
                        None
                    else:
                        # approximation of Laplace smoothing
                        p -= np.log(self.V)
        return np.exp(p)

    def compute_perplexity(self, data):
        """
        For a given data set of sentences s, compute the perplexity using the exp-log trick
        """
        perplexity = 0
        self.probas = []
        T = 0
        count_null_proba = 0
        for sentence in data.text:
            T += len(sentence.split())-2
            proba = self.compute_sentence_probability(sentence)
            self.probas.append(proba)
            if proba == 0:
                count_null_proba += 1
            else:
                perplexity -= np.log(proba)
        # perplexity += T * np.log(sum(self.probas)) normalize ?
        self.verboseprint('{} null sentence probabilities found in perplexity.'.format(count_null_proba))
        self.verboseprint('Done. Time elapsed: {:3.1f} sec.'.format(time()-self.t0))
        return np.exp(perplexity / T)

    
def print_pp(ng, train, valid, test):
    """
    Print perplexities on train, validation and test sets given a n-gram model ng
    """
    print()
    print('Perplexity on train set: {:.2f}'.format(ng.compute_perplexity(train)))
    print('Perplexity on val set: {:.2f}'.format(ng.compute_perplexity(valid)))
    print('Perplexity on test set: {:.2f}'.format(ng.compute_perplexity(test)))
        
    
def generate_next_word(seed_context, models):
    """
    Generate the next word given a context 'seed_context'.
    Models must be of type [ng(k), ng(k-1), ..., ng(1)] where ng(k) is a k-gram model (with interpolation,
    smoothing or backoff). Hence ng(1) is a unigram model. 
    
    :param seed_contexts: string, first words
    :param models: list of n-gram models
    
    Example: 
        seed_context = 'I am happy'
        - generate_next_word will try to find, in the 4-gram models ng(4), all contexts (of size 3) that
        start with 'I am happy', then pick the corresponding 4-grams and take the highest probability.
        It thus give the next word. If 'I am happy' can not be found, then, the same approach is done with
        'am happy' with the trigram model ng(3), etc... 
        If 'happy' is not a context of any bigram (which is not true here), then a random word is choosen
        in the unigram.
        - if the model outputs <s> as next word, then it falls back into the loop in order to find another word.
        - if the model outputs </s> as next word, then it stops.
    """
    context = seed_context
    k = len(context.split()) # number of words in context
    n = len(models) # number of models (i.e. maximal size of context - 1: ng(5)-> 5-gram model-> 4 words contexts)
    if k >= n:
        # crop the context with the maximal size of context than can be treated by the first model (ng(k))
        context = utils.split_and_join(context, n=k+1, n1=k-n+1)
    k = len(context.split())
    for j, model in enumerate(models[len(models)-k-1:-1]):
        contexts = model.context_count
        if context in contexts:
            # ngrams store which ngrams have context as contexts
            ngrams = [ngram for ngram in model.ngram_count.keys() if utils.split_and_join(ngram, model.n-1) == context]
            # ngrams is a dict {ngram: prob} with restricted ngrams (see above)
            ngrams = {ngram: prob for (ngram, prob) in model.probabilities.items() if ngram in ngrams}
            # next_word is the last word of the ngram with the highest probability
            next_word = utils.sort_dict(ngrams)[0][0].split()[-1]
            if next_word == '<s>':
                context = utils.split_and_join(context, model.n+1, 1)
            else:
                return next_word
        else: 
            # if k-gram can not be found, then loop and shift to k-1
            context = utils.split_and_join(context, model.n+1, 1)
    # if any context of any size can be found, pick a random words in model[-1] (i.e. unigrams)
    next_random_word = random.choice(models[-1].ngrams)
    while next_random_word == '<s>':
        next_random_word = random.choice(models[-1].ngrams)
    return next_random_word

def generate_sentence(seed_words, n_next, models):
    """
    Generate the n_next words of a sentence based on input words seed_words
    """
    k = len(seed_words.split())
    n = len(models)
    n1 = 0 if k-n < 0 else k-n+1
    context = utils.split_and_join(seed_words, n=k+1)
    for _ in range(n_next):
        next_word = generate_next_word(context, models)
        context += ' ' + next_word
        if next_word == '</s>':
            break
    return context



