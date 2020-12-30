import csv
import pickle
import numpy as np

from nltk import word_tokenize
from scipy.stats import ranksums
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from statistics import mean, stdev
from math import sqrt

import statsmodels.api as sm

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn import metrics
from random import sample


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
        with open('pickle/' + name + '.pkl', 'rb') as fout:
            return pickle.load(fout)
        # end with
    # end def

# end class


def extract_vad_dictionary(filename):
    vdict = dict(); adict = dict(); ddict = dict()
    with open(filename, 'r') as fin:
        for line in fin:
            if len(line.split('\t')) < 4: continue
            tokens = line.strip().split('\t')
            vdict[tokens[0]] = float(tokens[1])
            adict[tokens[0]] = float(tokens[2])
            ddict[tokens[0]] = float(tokens[3])
        # end for
    # end with

    Serialization.save_obj(vdict, 'v.dict')
    Serialization.save_obj(adict, 'a.dict')
    Serialization.save_obj(ddict, 'd.dict')

    print('saved dictionaries of', len(vdict), len(adict), len(ddict), 'length')

# end def


def compute_metric_for_definition(definition, d, avg):
    values = list()
    for token in definition:
        if token not in d: continue
        values.append(d[token])
    # end for

    if len(values) == 0: return -1
    '''
    if min(values) > avg: return max(values)-avg
    if max(values) < avg: return avg-min(values)
    if min(values) <= avg <= max(values): return max(values)-min(values)
    '''
    return np.mean(values)

# end def


def assign_vad_metrics_to_idioms_naive(filename):
    v_dict = Serialization.load_obj('v.dict'); mean_v = np.mean(list(v_dict.values()))
    a_dict = Serialization.load_obj('a.dict'); mean_a = np.mean(list(a_dict.values()))
    d_dict = Serialization.load_obj('d.dict'); mean_d = np.mean(list(d_dict.values()))
    print('means:', mean_v, mean_a, mean_d)

    with open(filename, 'r') as fin, open(filename.replace('csv', 'vad.naive.csv'), 'w') as fout:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for line in csv_reader:
            definition = word_tokenize(line[1].lower().strip())
            v_value = compute_metric_for_definition(definition, v_dict, mean_v)
            a_value = compute_metric_for_definition(definition, a_dict, mean_a)
            d_value = compute_metric_for_definition(definition, d_dict, mean_d)
            csv_writer.writerow(line + [v_value, a_value, d_value])
            print('done', line[0].strip())

        # end for
    # end with

# end def


def compute_lexicon_embeddings(model):
    word2embedding = dict()
    words = list(Serialization.load_obj('v.dict').keys())
    results = model.encode(words)

    assert(len(words) == len(results))
    for word, emb in zip(words, results): word2embedding[word] = emb
    Serialization.save_obj(word2embedding, 'lexicon2embeddings')

    print('saved', len(word2embedding), 'word embeddings...')

# end def


def compute_idiom_embeddings(filename, model):
    defs2embedding = dict()
    with open(filename, 'r') as fin:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        defs = [line[1].strip() for line in csv_reader]
    # end with

    results = model.encode(defs)
    assert(len(defs) == len(results))
    for definition, emb in zip(defs, results): defs2embedding[definition] = emb
    Serialization.save_obj(defs2embedding, 'definition2embeddings')

    print('saved', len(defs2embedding), 'definition embeddings...')

# end def


def assign_vad_metrics_to_idioms_transformers(filename):
    # todo: choose the model that maximizes goodness of fit
    # todo: from https://github.com/UKPLab/sentence-transformers
    model = SentenceTransformer('bert-large-nli-mean-tokens')
    compute_idiom_embeddings(filename, model)
    #compute_lexicon_embeddings(model)

# end def


def extract_vad_by_gender(filename):
    idiom2vad = dict()
    with open(filename, 'r') as fin:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)  # skip the header
        for line in csv_reader:
            idiom2vad[line[0].strip()] = [float(line[6]), float(line[7]), float(line[8])]
        # end for
    # end with
    print('loaded', len(idiom2vad), 'vad values for idioms')

    m_vad = list(); f_vad = list()
    with open(filename, 'r') as fin:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)  # skip the header
        for line in csv_reader:
            m_count = int(line[3]); f_count = int(line[4])
            vad = idiom2vad.get(line[0].strip(), None)

            if line[0].strip() in excluded: continue
            '''
            if vad is None or not 0.25 < vad[VAD_INDEX] < 0.75:
                #print('not found vad for', line[0].strip(), '&', line[4])
                continue
            # end if
            '''

            for i in range(m_count): m_vad.append(vad)
            for i in range(f_count): f_vad.append(vad)
            #m_vad.append([m_count*i for i in vad])
            #f_vad.append([f_count*i for i in vad])

        # end for

    # end with

    m_vad = sample(m_vad, SAMPLE_SIZE); f_vad = sample(f_vad, SAMPLE_SIZE)
    m_vad_array = np.array(m_vad); f_vad_array = np.array(f_vad)
    print(m_vad_array.shape, f_vad_array.shape)

    m_metric = m_vad_array[:, VAD_INDEX]; f_metric = f_vad_array[:, VAD_INDEX]
    print(len(m_metric), len(f_metric), np.mean(m_metric), np.std(m_metric), np.mean(f_metric), np.std(f_metric))
    print(ranksums(m_metric, f_metric))

    cohens_d = (mean(m_metric) - mean(f_metric)) / (sqrt((stdev(m_metric) ** 2 + stdev(f_metric) ** 2) / 2))
    #cohens_d = (mean(m_metric) - mean(f_metric)) / stdev(m_metric + f_metric)
    print('cohens d:', cohens_d)

    #Serialization.save_obj(m_metric, 'm.D')
    #Serialization.save_obj(f_metric, 'f.D')

# end def


def extract_sentiment_by_gender(f_sentiment, f_counts):
    idiom2vad = dict()
    with open(f_sentiment, 'r') as fin:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in csv_reader: idiom2vad[line[0].strip()] = [int(line[2])]
    # end with
    print('loaded', len(idiom2vad), 'sentiment values for idioms')

    m_sentiment = list(); f_sentiment = list()
    with open(f_counts, 'r') as fin:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)  # skip the header
        for line in csv_reader:
            m_count = int(line[3]); f_count = int(line[4])
            sentiment = idiom2vad.get(line[0].strip(), None)
            if sentiment is None:
                print('not found sentiment for', line[0].strip(), '&', line[5])
                continue
            # end if

            #sentiment = float(line[11])
            m_sentiment.extend(m_count*[sentiment])
            f_sentiment.extend(f_count*[sentiment])
        # end for

    # end with

    m_metric = m_sentiment; f_metric = f_sentiment
    print(np.mean(m_metric), np.mean(f_metric), len(m_metric), len(f_metric))
    print(ranksums(m_metric, f_metric))

# end def


def regress_definitions_embeddings(w2e, d2e, vad_dict, type):
    import random

    x_test = list(); y_test = list()
    x_train = list(); y_train = list()
    test_set = random.sample(range(len(w2e)), 1000)
    for i, word in enumerate(list(w2e.keys())):
        if i in test_set:
            x_test.append(w2e[word])
            y_test.append(vad_dict[word])
            continue
        # end if
        x_train.append(w2e[word])
        y_train.append(vad_dict[word])
    # end for

    x_defs = list(); defs = list()
    for definition in d2e:
        x_defs.append(d2e[definition])
        defs.append(definition)
    # end for

    '''
    # supposed to work as beta regression for proportion dependent
    # https://danvatterott.com/blog/2018/05/03/regression-of-a-proportion-in-python/
    binom_glm = sm.GLM(y_train, x_train, family=sm.families.Binomial())
    binom_model = binom_glm.fit()  # fit with train data
    '''

    binom_model = Serialization.load_obj('binom.model.' + type)
    y_predict = binom_model.predict(x_train)  # how good is the fit of train data
    print('r^2 on train data:', metrics.r2_score(y_train, y_predict))
    y_predict = binom_model.predict(x_test)

    assert(len(y_test) == len(y_predict))
    print('corr to held-out set:', pearsonr(y_test, y_predict))

    y_defs_predict = binom_model.predict(x_defs)

    d2v = dict()
    for definition, val in zip(defs, y_defs_predict):
        assert(0.0 <= val <= 1.0), 'predicted value is not proportion'
        d2v[definition] = val
    # end for
    return d2v

# end def


def compute_r2_score(y_true, y_predicted):
    ybar = np.mean(y_true)
    ssreg = np.sum((np.array(y_predicted) - np.array(y_true)) ** 2)
    sstot = np.sum((np.array(y_true) - ybar) ** 2)
    return 1.0 - float(ssreg) / sstot

# end def


def infer_definitions_vad(filename):
    w2e = Serialization.load_obj('lexicon2embeddings')
    d2e = Serialization.load_obj('definition2embeddings')

    d2v = regress_definitions_embeddings(w2e, d2e, Serialization.load_obj('v.dict'), 'v')
    d2a = regress_definitions_embeddings(w2e, d2e, Serialization.load_obj('a.dict'), 'a')
    d2d = regress_definitions_embeddings(w2e, d2e, Serialization.load_obj('d.dict'), 'd')

    '''
    with open(filename, 'r') as fin, open(filename.replace('csv', 'vad-embeddings.csv'), 'w') as fout:
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


def enrich_relationships_data_with_vad(filename):
    idiom2vad = dict()
    with open('idioms-definitions-final-counts.vad-embeddings.csv', 'r') as fin:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)  # skip the header
        for line in csv_reader:
            idiom2vad[line[0].strip()] = [float(line[6]), float(line[7]), float(line[8])]
        # end for
    # end with
    print('loaded', len(idiom2vad), 'vad values for idioms')

    with open(filename, 'r') as fin, open(filename.replace('csv', 'vad-embeddings.csv'), 'w') as fout:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)
        csv_writer.writerow(header[0:1] + ['', ''] + header[1:] + ['V', 'A', 'D'])  # skip the header
        for line in csv_reader:
            idiom = line[0].strip()
            csv_writer.writerow(line[0:1] + ['', ''] + line[1:] + idiom2vad.get(idiom, []))
        # end for
    # end with

# end def


def evaluate_regression_results():
    w2e = Serialization.load_obj('lexicon2embeddings')
    d2e = Serialization.load_obj('definition2embeddings')
    definition = 'to argue or fight with someone about something'
    print(cosine(d2e[definition], w2e['angry']))
    print(cosine(d2e[definition], w2e['happy']))

# end def


VAD_INDEX = 2  # 0-2
SAMPLE_SIZE = 320000
excluded = ['red herring', 'red carpet']
if __name__ == '__main__':

    #extract_vad_dictionary('nrc-vad-lexicon.tsv')
    #assign_vad_metrics_to_idioms_naive('idioms-definitions.csv')

    #assign_vad_metrics_to_idioms_transformers('idioms-definitions-final-counts.csv')
    infer_definitions_vad('idioms-definitions-final-counts.csv')

    #extract_vad_by_gender('idioms-definitions-final-counts.vad-embeddings.csv')
    #extract_sentiment_by_gender('idioms-definitions.seniment.csv',
    #    'idioms-definitions-final-counts.vad-embeddings.csv')

    # handle relationships
    #enrich_relationships_data_with_vad('idioms-definitions-final-counts.rel.csv')
    #extract_vad_by_gender('idioms-definitions-final-counts.rel.vad-embeddings.csv')

    # todo: work with emotion in addition to vad
    #evaluate_regression_results()


# end if
