import pandas as pd
import numpy as np
import os
import io
from tqdm import tqdm
import re
import string
import numpy as np
from time import time
import pickle as pkl
from copy import deepcopy
from collections import Counter
import operator
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model, model_from_yaml

from keras import backend as K

############################################## N-grams based modelling ##############################################

def read_data(name):
    """
    Read the data
    :param name: str, exact name of data set to read
    :return: pd.DataFrame with 1 column containing the sentences
    """
    d = pd.read_csv(open(os.path.join('./Data/', name), 'r'), sep=" .\n", header=None,
                    engine='python')
    d.columns = ['text']
    if 'char' in name:
        print('Cleaning to redefine (character-level data set)')
    return d

def read_split(name):
    """
    Read train, test, validation sets
    :param name: str, choose between 'wiki', 'char', 'tatoeba'
    """
    if name not in ['wiki', 'char', 'tatoeba']:
        raise AssertionError("Please choose between: 'wiki', 'char' or 'tatoeba'")
    train, test, valid = [read_data(name + '.' + d + '.txt') for d in ['train', 'test', 'valid']]
    print('Loading ' + name + ' data set...')
    print('Number of training sentences:   {:10.0f}'.format(train.shape[0]))
    print('Number of validation sentences: {:10.0f}'.format(valid.shape[0]))
    print('Number of testing sentences:    {:10.0f}'.format(test.shape[0]))
    print('Done.')
    return train, test, valid
    
words_replace = dict()
for char, sub in zip([
        "wasn 't", "isn 't",
        "you' re", "we' re", "they' re",
        "can 't",
        "don 't'", "doesn 't", "didn 't",
        "haven' t", "hasn 't'",
        "shouldn 't", "wouldn 't",  "won 't",
        "' ll",
        "I' m"
    ],
        ['was not', 'is not',
         'you are', 'we are', 'they are',
         'can not',
         'do not', 'does not', 'did not',
         'have not', 'has not',
         'should not', 'would not', 'will not',
         ' will',
         'I am'
        ]):
    words_replace[char] = sub

flatten = lambda l: [item for sublist in l for item in sublist]

def ngram_nltk(s, n):
        """
        For a given sentence s, return list of n-grams
        """
        # Break sentence in the token, remove empty tokens
        tokens = [token for token in s.split(" ") if token != ""]
        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
        ngrams = zip(*[tokens[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]
    
punct_replace = dict({char: ' ' for char in list(string.punctuation)})
dict_replace = {**words_replace, **punct_replace}

def clean_text(df):
    """
    Clean sentences by:
        - lowering characters
        - removing non alphabetic characters except brackets ('<unk>')
        - adding start and end-symbol (resp. <s> and </s>)
    """
    data = deepcopy(df)
    data.text = data.text.apply(lambda s: s[:-2].lower())
    data.text = data.text.apply(lambda s: s.replace('<unk>', '<UNK>'))
    for item in dict_replace.items():
        char, sub = item
        data.text = data.text.apply(lambda s: s.replace(char, sub))
    data.text = data.text.apply(lambda s: re.sub(r'[^a-zA-Z0-9\s]', ' ', s))
    data.text = data.text.apply(lambda s: re.sub(' +',' ', s))
    data.text = data.text.apply(lambda s: s.replace('UNK', '<unk>'))
    data.text = data.text.apply(lambda s: '<s> '+s+' </s>')
    return data


def words_count(data):
    """
    Print some statistics about the data:
        - Total number of words
        - Number of unique words
        - Percentage of OOV
    """
    split = data.text.apply(lambda s: s.split())
    words = dict(Counter(flatten(list(split))))
    words_count = sum(words.values())
    unique_count = len(set(words.keys()))
    unk_count = words['<unk>']
    p_unk = unk_count/words_count * 100
    print('Total number of words: {:5.0f}\nNumber of unique words: {:3.0f}\nPercentage of OOV: {:2.3f}%'.format(words_count,
                                                                                                               unique_count,
                                                                                                               p_unk))
    return words

def split_and_join(s, n, n1=0):
    """
    For a given sentence s = 'I am not a genius', n=2 returns 'I am', n=3 returns 'I am not', etc...
    """
    s = s.split()[n1:n]
    return " ".join(s)

def sort_dict(dict_):
    """
    Sort a dictionary by key
    """
    return sorted(dict_.items(), key=operator.itemgetter(1))[::-1]


############################################## Deep Learning based modelling ##############################################


def load_vectors(fname, vocab, nmax=2*10**6):
    """
    Load pre-trained words embeddings from FastText:
    A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, 'FastText.zip: Compressing text classification models'
    """
    word2vec = {}
    with io.open(fname, encoding='utf-8') as f:
        next(f)
        for i, line in tqdm(enumerate(f), total=nmax):
            word, vec = line.split(' ', 1)
            if word in vocab:
                word2vec[word] = np.fromstring(vec, sep=' ')
            if i == (nmax - 1):
                break
    print('Loaded %s pretrained word vectors' % (len(word2vec)))
    return word2vec


class keras_preprocess():
    """
    Preprocess the training, validation and testing data sets. 
    Returns: np.arrays, X_train, X_valid, X_test, y_train, y_valid and y_test
    """
    def __init__(self, train, valid, test, ngram, verbose=True):
        """
        If ngram = 3, then X.shape = (number of trigrams, 2), y.shape = (number of trigrams, 1)
        X contains the 2 first words, y the last one.
        """
        self.t0 = time()
        self.train = train
        self.valid = valid
        self.test = test
        self.ngram = ngram
        self.verboseprint = print if verbose else lambda *a, **k: None
        self.verboseprint('Tokenizing...')
        self.tokenize()
        self.verboseprint('Creating n-grams...')
        self.sequences_train = self.create_ngrams(self.train)
        self.sequences_valid = self.create_ngrams(self.valid)
        self.sequences_test = self.create_ngrams(self.test)
        
    def tokenize(self):
        """
        Fit tokenizer
        """
        # Lower, remove non alphabetic characters except brackets and antislash
        self.tokenizer = Tokenizer(num_words=None,
                              filters='!"#$%&()*+,-.:;=?@^_`{|}~\t\n',
                              lower=True,
                              split=' ',
                              char_level=False,
                              oov_token='<unk>',
                              document_count=0)
        # Fit on train
        self.tokenizer.fit_on_texts(self.train.text)
        self.vocab = self.tokenizer.word_index
        self.vocab_size = len(self.vocab)+2
        
    def create_ngrams(self, data):
        """
        Return sequences (ngrams). If self.ngram=3, then output.shape = (number of sequences, 3)
        """
        sequences = list()
        for line in data.text:
            encoded = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(self.ngram, len(encoded)):
                sequence = encoded[i-self.ngram:i]
                sequences.append(sequence)
        return np.stack(sequences, axis=0)
    
    def split_X_y(self, sequences):
        """
        Split sequences into X and y sets (X: first self.ngram-1 words, y: last words)
        """
        X, y = sequences[:, :-1], sequences[:, -1]
        #y = to_categorical(y, num_classes=self.vocab_size)
        return X, y
    
    def output(self):
        """
        Store the quantities of interest
        """
        self.verboseprint('Split into X and y...')
        self.X_train, self.y_train = self.split_X_y(self.sequences_train)
        self.X_valid, self.y_valid = self.split_X_y(self.sequences_valid)
        self.X_test, self.y_test = self.split_X_y(self.sequences_test)
        self.verboseprint('Done. Time elapsed: {:3.1f} sec.'.format(time()-self.t0))
        return self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test    
    
    
def keras_print_score(X, y, model, dataset):
    """
    Print metrics for a given tuple (X, y)
    dataset: string, 'Train', 'Test' or 'Valid'
    """
    l, p, a = model.evaluate(X, y, verbose=0)
    print(dataset + ' set - Loss: {:.3f} | Perplexity: {:.2f} | Accuracy: {:3f}'.format(l, p, a))
    return l, p, a
    
def keras_perplexity(y_true, y_pred):
    """
    Compute perplexity using keras.backend
    Source: https://nbviewer.jupyter.org/github/chakki-works/chariot/blob/master/notebooks/language%20modeling.ipynb
    """
    cross_entropy = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
    perplexity = K.exp(cross_entropy)
    return perplexity

def keras_generate_seq(model, tokenizer, max_length, seed_text, n_words):
    """
    Generate sequence based on output probabilities
    :param model: keras.engine.sequential.Sequential instance
    :param tokenizer: keras_preprocessing.text.Tokenizer instance
    :param max_length: int, size of ngrams 
    :param seed_text: string, first words of the sentence
    :param n_words: int, number of words to predict
    """
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # pre-pad sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
    return in_text

def keras_load_model(modelname):
    '''
    Load model from modelname
    '''
    # load YAML
    print('Loading model...')
    yaml_file = open(os.path.join('./Models', modelname+'.yaml'), 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(os.path.join('./Models/', modelname+".h5"))
    print("Loaded model from disk.")
    return loaded_model 

def load_results(layer, dataset='tatoeba', best=False):
    """
    For a given recurrent layer, store and return all results obtained from the experiments
    :param layer: string
    :param dataset: string, choose between tatoeba or wiki
    :param best: Boolean, if comparison of best models
    """
    layer_ = '_'+layer+'_'
    path = './Results/'
    path_ = path if not best else './Results/best'
    entries = []
    for entry in os.scandir(path):
        condition = ('best' in entry.path) if best else ('best' not in entry.path)
        if (layer_ in entry.path) and condition and (dataset in entry.path):
            entries.append(entry.path)
    n = len(entries)
    zo = np.zeros(n+1)
    print('Loading %d results for %s...' %(n, layer))
    # Then create dataframe
    results = pd.DataFrame({'dataset': zo,'layer': zo,'ngram': zo, 'embedding': zo, 'mcells': zo, 'batchsize': zo, 'dropout2': zo,
                           'loss_train': zo, 'per_train': zo, 'acc_train': zo,
                           'loss_val': zo, 'per_val': zo, 'acc_val': zo,
                           'loss_test': zo, 'per_test': zo, 'acc_test': zo})
    # Finally, fill it
    j = 0
    for entry in os.scandir(path):
        condition = ('best' in entry.path) if best else ('best' not in entry.path)
        if (layer_ in entry.path) and condition and (dataset in entry.path):
            j += 1
            params = entry.path.split('.pk')[0].split(path_)[1].split('_')[1*best:]
            dataset = params[0]
            layer = params[1]
            ngram = int(params[2][5:])
            emb = int(params[3][3:])
            mcells = int(params[4][6:])
            batchsize = int(params[5][5:])
            dropout2 = int(params[7][-1]) * 0.1
            r = list(pkl.load(open(entry.path, 'rb')).values())
            results.iloc[j, :7] = [dataset, layer, ngram, emb, mcells, batchsize, dropout2]
            results.iloc[j, 7:] = r
    results = results[results['dataset'] != 0]
    return results

def plot_comp(results, param, layer, metric):
    """
    Plot the mean metric values as a function of a given parameter, for both train, test and val sets
    along with error bars.
    :param results: pd.Dataframe
    :param param: string, choose between 'ngram', 'embedding', 'mcells' or 'dropout2'
    :param layer: string, 'lstm' or 'bilstm', etc... (layer only serves for the title)
    :param metric: string, choose between 'per' (perplexity), 'acc' (accuracy) or 'loss' (cross-entropy)
    """
    # First we apply groupby + mean or + std
    if 'per' in metric:
        rm = results.groupby(param).mean()[['per_train', 'per_val', 'per_test']]
        rstd = results.groupby(param).std()[['per_train', 'per_val', 'per_test']]
    elif 'acc' in metric:
        rm = results.groupby(param).mean()[['acc_train', 'acc_val', 'acc_test']]
        rstd = results.groupby(param).std()[['acc_train', 'acc_val', 'acc_test']]
    else:
        rm = results.groupby(param).mean()[['loss_train', 'loss_val', 'loss_test']]
        rstd = results.groupby(param).std()[['loss_train', 'loss_val', 'loss_test']]
    # Then we draw the errorbars
    plt.errorbar(x=rm.index, y=rm[metric+'_'+'train'], yerr=rstd[metric+'_'+'train'], marker='o', label='Train')
    plt.errorbar(x=rm.index, y=rm[metric+'_'+'val'], yerr=rstd[metric+'_'+'val'], marker='o', label='Validation')
    plt.errorbar(x=rm.index, y=rm[metric+'_'+'test'], yerr=rstd[metric+'_'+'test'], marker='o', label='Test')
    plt.legend(loc='best')
    # Change metric name for the title
    metric_ = 'Accuracy' if metric=='acc' else 'Perplexity' if metric=='per' else 'Entropy'
    plt.title('%s as a function of %s (%s)' % (metric_, param, layer))
    # Be sure to convert into 
    plt.xticks(rm.index, np.round(rm.index, 1))