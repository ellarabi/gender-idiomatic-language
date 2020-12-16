import sys
import csv
import pickle, random
import numpy as np

from nltk import sent_tokenize
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from scipy.stats import ranksums
from statistics import mean, stdev
from math import sqrt


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


def get_random_posts(filename):
    count = 0
    posts = list()
    with open(filename, 'r') as fin:
        for line in fin:
            posts.append(line.strip())
            if count > 500000: break
            count += 1
        # end for
    # end with

    return random.sample(posts, SAMPLE_SIZE)

# end def


def vad_analysis_background(type, count):
    posts_f = get_random_posts(dirname + 'data.clean.F.dat')
    posts_m = get_random_posts(dirname + 'data.clean.M.dat')
    print('sampled', SAMPLE_SIZE, 'posts from m and f data ...')

    model = SentenceTransformer('bert-large-nli-mean-tokens')
    regressor = Serialization.load_obj('binom.model.' + type)
    print('uploaded models for', type.capitalize(), '...')

    words = list(Serialization.load_obj(type+'.dict').keys())

    print('computing m scores ...')
    scores_m = extract_scores(posts_m, model, regressor, words)
    print('computing f scores ...')
    scores_f = extract_scores(posts_f, model, regressor, words)

    Serialization.save_obj(scores_m, 'scores.m.' + type + '.' + str(count))
    Serialization.save_obj(scores_f, 'scores.f.' + type + '.' + str(count))

    print('finished')

# end def


def infer_emotion_value(embeddings, regressor):
    predictions = regressor.predict(embeddings)
    assert(len(predictions) == len(embeddings))
    return predictions

# end def


def extract_scores(data, model, regressor, words):
    result = list()
    for i, post in enumerate(data):
        sentences = get_sentences(post, words)
        if len(sentences) == 0: continue
        embeddings = [emb for emb in model.encode(sentences)]
        values = infer_emotion_value(embeddings, regressor)  # predict vad dimension
        result.append(np.mean(values))

        if i % 100 == 0:
            print(i); sys.stdout.flush()
        # end if
    # end for

    return result
# end def


def test_vad_differences(type, count):
    scores_m = Serialization.load_obj('scores.m.' + type + '.' + str(count))
    scores_f = Serialization.load_obj('scores.f.' + type + '.' + str(count))
    cohens_d = (mean(scores_m) - mean(scores_f)) / (sqrt((stdev(scores_m) ** 2 + stdev(scores_f) ** 2) / 2))
    #cohens_d = (mean(scores_m) - mean(scores_f)) / stdev(scores_m + scores_f)

    scores = scores_f + scores_m
    print('total population stats:', np.mean(scores), np.std(scores))
    print(np.mean(scores_m), np.std(scores_m), np.mean(scores_f), np.std(scores_f))
    print(ranksums(scores_m, scores_f))
    print('cohens d:', cohens_d)

# end def


def get_sentences(text, words):
    sentences = list()
    for s in sent_tokenize(text):
        if len(s.split()) <= 2: continue
        if len(set(s.lower().split()).intersection(set(words))) < 2: continue
        sentences.append(s)
    # end for
    return sentences
# end def


format = '{:.3f}'
VAD_INDEX = 2  # 0-2
SAMPLE_SIZE = 20000
dirname = '<working dir>/gender-idioms/'

if __name__ == '__main__':

    #vad_analysis_background('d', 3)
    test_vad_differences('d', 3)


# end if
