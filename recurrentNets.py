import os
import numpy as np
from copy import deepcopy
import pickle as pkl
import matplotlib.pyplot as plt
from utils import keras_perplexity, keras_print_score
from sklearn.linear_model import LogisticRegression

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.layers import Embedding, Dense, LSTM, CuDNNLSTM, SimpleRNN, GRU, CuDNNGRU, Bidirectional, Activation, Flatten, Reshape, Dropout, Conv2D, MaxPooling2D, Input
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras import backend as K


class LrReducer(Callback):
    """
    Learning Rate Scheduler. If the monitored metric is not improving, reduce the learning 
    rate by reduce_rate. This can be done for a maximum of reduce_nb times. 
    """
    def __init__(self, patience=0, reduce_rate=2, reduce_nb=10, monitor='val_keras_perplexity', verbose=1):
        super(Callback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_score = 10**10 if 'perplexity' in monitor else -1.
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.monitor = monitor
        self.verbose = verbose
        self.sup = lambda x,y: x > y
        self.inf = lambda x,y: x < y

    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get(self.monitor)
        comp = self.inf if 'perplexity' in self.monitor else self.sup
        if comp(current_score, self.best_score):
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                print('---current best val metric: %.3f' % current_score)
        else:
            if self.wait >= self.patience:
                self.current_reduce_nb += 1
                if self.current_reduce_nb <= 10:
                    lr = K.get_value(self.model.optimizer.lr)
                    K.set_value(self.model.optimizer.lr, lr/self.reduce_rate)
                    if self.verbose > 0:
                        print('---descreased lr=%E to lr=%E' % (lr, lr/self.reduce_rate))
                else:
                    if self.verbose > 0:
                        print("Epoch %d: early stopping" % (epoch))
                    self.model.stop_training = True
            self.wait += 1
            

class RecDeepNetworks():
    """
    Implementation of Long-Short Term Memory Networks using Keras
    """
    def __init__(self, preprocessing, modelname, verbose=0):
        """
        :param preprocessing: utils.keras_preprocess instance
        :param modelanme: string, name to use for storing weights, results, etc...
        """
        self.preprocessing = preprocessing
        self.modelname = modelname
        self.verboseprint = print if verbose==0 else lambda *a, **k: None
        self.gpu = len(K.tensorflow_backend._get_available_gpus())>0
        self.layers = {'lstm': LSTM,
                       'cudnnlstm': CuDNNLSTM,
                       'rnn': SimpleRNN,
                       'gru': GRU,
                       'cudnngru': CuDNNGRU}
        
    def _init(self, recurrent_layer, pre_trained, embedding, memorycells, dropout_rate1, dropout_rate2, **kwargs):
        """
        :param recurrent_layer: string, type of reccurent filter
        :param embedding: int, dimension of embedding space
        :param memorycells: int, number of memory cells in LSTM
        :param dropout_rate: int, dropout rate before applying softmax
        :param **kwargs: for reccurent layers: dropout, recurrent_dropout (don't work with Cudnn layers)
        """
        if recurrent_layer not in ['ffnn', 'lstm', 'cudnnlstm', 'bilstm', 'cudnnbilstm', 'bilstmconv2D', 'rnn', 'gru', 'cudnngru']:
            raise AssertionError("Please choose between: 'ffnn', lstm','cudnnlstm', 'bilstmconv2D', 'bilstm', 'cudnnbilstm', 'rnn', 'gru' or 'cudnngru'")
            
        if pre_trained:
            # create a weight matrix for words in training sentences
            self.embedding_matrix = np.random.uniform(-0.1, 0.1, size=(self.preprocessing.vocab_size, 300))
            for i, word in enumerate(self.preprocessing.vocab.keys()):
                try:
                    self.embedding_matrix[i] = self.preprocessing.word2vec[word]
                except:
                    pass
            embedding_layer = Embedding(self.preprocessing.vocab_size, 300, input_length=self.preprocessing.ngram-1,
                                        weights=[self.embedding_matrix], trainable=False)
        else:
            embedding_layer = Embedding(self.preprocessing.vocab_size, embedding, input_length=self.preprocessing.ngram-1)
        
        if recurrent_layer in ['lstm', 'cudnnlstm', 'rnn', 'gru', 'cudnngru']:
            # LSTM (cpu or gpu); RNN (cpu only); GRU (cpu or gpu)
            self.model = Sequential()
            self.model.add(embedding_layer)
            self.model.add(Dropout(dropout_rate1))
            layer = self.layers[recurrent_layer](units=memorycells, **kwargs)
            self.model.add(layer)
            self.model.add(Dropout(dropout_rate2))
            self.model.add(Dense(self.preprocessing.vocab_size, activation='softmax'))
             
        elif recurrent_layer in ['bilstm', 'cudnnbilstm']:
            # BiLSTM (cpu or gpu)
            self.model = Sequential()
            self.model.add(embedding_layer)
            self.model.add(Dropout(dropout_rate1))
            if recurrent_layer == 'bilstm':
                layer = self.layers['lstm'](units=memorycells, **kwargs)
            else:
                layer = self.layers['cudnnlstm'](units=memorycells, **kwargs)
            self.model.add(Bidirectional(layer, merge_mode='concat'))
            self.model.add(Dropout(dropout_rate2))
            self.model.add(Dense(self.preprocessing.vocab_size, activation='softmax'))
            
        elif recurrent_layer == 'ffnn':
            # Feed Forward NN (no recurrent layer)
            self.model = Sequential()
            self.model.add(embedding_layer)
            self.model.add(Flatten())
            self.model.add(Activation('sigmoid'))
            self.model.add(Dropout(dropout_rate2))
            self.model.add(Dense(self.preprocessing.vocab_size, activation='softmax'))
            
        elif recurrent_layer == 'bilstmconv2D':
            # Implementation of BLSTM2DCNN : Peng et al. (2016),
            # "Text Classification Improved by Integrating Bidirectional LSTM with Two-dimensional Max Pooling"
            sequence_input = Input(shape=(self.preprocessing.ngram-1,), dtype='int32')
            embedded_sequences = embedding_layer(sequence_input)
            embedded_sequences = Dropout(dropout_rate1)(embedded_sequences)
            if self.gpu:
                x = Bidirectional(CuDNNLSTM(units=memorycells, return_sequences=True, **kwargs))(embedded_sequences)
            else:
                x = Bidirectional(LSTM(units=memorycells, return_sequences=True, **kwargs))(embedded_sequences)
            x = Reshape((2*(self.preprocessing.ngram-1), memorycells, 1))(x)
            x = Conv2D(64, (3, 3))(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(dropout_rate2)(x)
            x = Flatten()(x)
            preds = Dense(self.preprocessing.vocab_size, activation='softmax')(x)
            self.model = Model(sequence_input, preds)
        self.verboseprint(self.model.summary())
        
    def _compile(self, loss='sparse_categorical_crossentropy', optimizer='adam'):
        """
        :param loss: string or keras.losses function
        :param optimizer: string of keras.optimizers function
        """
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[keras_perplexity, 'accuracy'])
        
    def _fit(self, batch_size, epochs, monitor, patience=5, verbose=1, **kwargs):
        """
        :param batch_size: int
        :param epochs: int, maximal number of epochs
        """
        if 'perplexity' in monitor:
            mode = 'min'
        elif 'acc' in monitor:
            mode = 'auto'
        self.model.fit(
            x=self.preprocessing.X_train,
            y=self.preprocessing.y_train,
            validation_data=(self.preprocessing.X_valid, self.preprocessing.y_valid),
            batch_size=batch_size, epochs=epochs, verbose=verbose,
            **kwargs
        )
                                  
    def _print_results(self):
        """
        Print metrics for both train, validation and test sets and pickle it in Results folder
        """
        loss_train, per_train, acc_train = keras_print_score(self.preprocessing.X_train, self.preprocessing.y_train, self.model, 'Train')
        loss_valid, per_valid, acc_valid = keras_print_score(self.preprocessing.X_valid, self.preprocessing.y_valid, self.model, 'Validation')
        loss_test, per_test, acc_test = keras_print_score(self.preprocessing.X_test, self.preprocessing.y_test, self.model, 'Test')
        results = {'loss_train': loss_train, 'per_train': per_train, 'acc_train': acc_train,
                   'loss_valid': loss_valid, 'per_valid': per_valid, 'acc_valid': acc_valid,
                   'loss_test': loss_test, 'per_test': per_test, 'acc_test': acc_test}
        pkl.dump(results, open(os.path.join('./Results',self.modelname+'.pk'), 'wb'))
                       
                       
    def _plot_results(self, modelname):
        """
        Plot perplexity and accuracy as a function of epochs (train + val)
        """
        h = self.model.history.history
        plt.figure(figsize=(15,5))
        plt.subplot(121)
        plt.plot(h['val_keras_perplexity'], label='Validation', marker='o')
        plt.plot(h['keras_perplexity'], label='Train', marker='o')
        plt.title(modelname+': perplexity with epochs')
        plt.legend()
        plt.subplot(122)
        plt.plot(h['val_acc'], label='Validation', marker='o')
        plt.plot(h['acc'], label='Train', marker='o')
        plt.title(modelname+': accuracy with epochs')
        plt.legend()
        plt.show()
        
    def _save(self):
        """
        Save model architecture and weights
        """
        # serialize model to YAML
        model_yaml = self.model.to_yaml()
        with open(os.path.join("./Models/", self.modelname+".yaml"), "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        self.model.save_weights(os.path.join("./Models/", self.modelname+".h5"))
        self.verboseprint("Saved model to disk.")