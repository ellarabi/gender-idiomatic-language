import csv
import pickle
from keras import Sequential
from sklearn import metrics
import numpy as np

from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping


class Serialization:
    @staticmethod
    def save_obj(obj, name):
        """
        serialization of an object
        :param obj: object to serialize
        :param name: file name to store the object
        """
        with open('pickle/' + name + '.pkl', 'wb') as fout:
            pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)
        # end with
    # end def

    @staticmethod
    def load_obj(name):
        """
        de-serialization of an object
        :param name: file name to load the object from
        """
        with open(name + '.pkl', 'rb') as fout:
            return pickle.load(fout)
        # end with
    # end def

# end class


def infer_definitions_vad(filename):
    w2e = Serialization.load_obj('../pickle/lexicon2embeddings')
    d2e = Serialization.load_obj('../pickle/definition2embeddings')

    d2v = extract_definitions_metrics(w2e, d2e, Serialization.load_obj('pickle/v.dict'), 'v')
    #d2a = extract_definitions_metrics(w2e, d2e, Serialization.load_obj('pickle/a.dict'), 'a')
    #d2d = extract_definitions_metrics(w2e, d2e, Serialization.load_obj('pickle/d.dict'), 'd')

    '''
    with open(filename, 'r') as fin, open(filename.replace('csv', 'vad-embeddings.nn.csv'), 'w') as fout:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(next(csv_reader) + ['V', 'A', 'D'])

        for line in csv_reader:
            definition = line[1].strip()
            csv_writer.writerow(line + [d2v[definition], d2a[definition], d2d[definition]])
        # end for

    # end with
    '''

# end def


def extract_train_test_data(w2e, d2e, vad_dict):
    x_train = list(); y_train = list()
    for word in w2e:
        x_train.append(w2e[word])
        y_train.append(vad_dict[word])
    # end for

    x_test = list(); defs = list()
    for definition in d2e:
        x_test.append(d2e[definition])
        defs.append(definition)
    # end for

    return x_train, y_train, x_test, defs
# end def


def extract_definitions_metrics(w2e, d2e, vad_dict, type):
    # https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33
    x_train, y_train, x_test, defs = extract_train_test_data(w2e, d2e, vad_dict)
    print('extracted train-test data:', len(x_train), len(y_train), len(x_test))

    '''
    print(np.array(x_train).shape)
    print('building and training regression nn model...')
    model = build_nn_model(np.array(x_train).shape, training=True)
    train_model(model, np.array(x_train), np.array(y_train))
    '''


    model = build_nn_model(np.array(x_train).shape, training=False)
    model.load_weights(dirname + 'model-weights-017--0.0934388.hdf4')  # load best model
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    y_predict = model.predict(np.array(x_train))  # how good is the fit of train data
    y_predict = [1.0 if y > 1.0 else 0 if y < 0.0 else y for y in y_predict]
    print('r^2 on train data:', metrics.r2_score(y_train, y_predict))


    #return extract_def_predictions(model, x_test, defs)

# end def


def extract_def_predictions(model, x_test, defs):
    d2v = dict()
    y_test_predict = model.predict(np.array(x_test))
    for definition, val in zip(defs, y_test_predict):
        assert(0.0 <= val <= 1.0), 'predicted value is not proportion'
        d2v[definition] = val
    # end for

    return d2v
# end def


def build_nn_model(dimensions, training=True):
    model = Sequential()
    print('building sequential model with input dim:', dimensions[1])
    model.add(Dense(512, kernel_initializer='normal', input_dim=dimensions[1], activation='relu'))
    if training: model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    if training: model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Dense(128,  kernel_initializer='normal', activation='relu'))
    if training: model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    model.summary()

    return model

# end def


def train_model(model, x, y):
    checkpoint_name = dirname + 'model-weights-{epoch:03d}--{val_loss:.7f}.hdf4'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    callbacks_list = [checkpoint, es]

    model.fit(x, y, epochs=300, batch_size=32, validation_split=0.2, callbacks=callbacks_list)

# end def


VAD_INDEX = 2  # 0-2
dirname = '<working dir>/gender-idioms/model-checkpoints/'
excluded = ['red herring', 'red carpet']
if __name__ == '__main__':

    infer_definitions_vad('../idioms-definitions-final-counts.csv')


# end if
