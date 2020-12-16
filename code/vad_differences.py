import csv
import pickle
import random
import numpy as np
from nltk import word_tokenize
from scipy.stats import ranksums


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


def read_definition_words(filename):
    with open(filename, 'r') as fin:
        csv_reader = csv.reader(fin, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)

        words = set()
        for line in csv_reader:
            definition = word_tokenize(line[1].strip().lower())
            for word in definition:
                if not word.isalpha(): continue
                words.add(word)
        # end for
    # end with

    return words

# end def


def read_concreteness_scores(filename):
    word2value = dict()
    with open(filename, 'r') as fin:
        csv_reader = csv.reader(fin, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)
        for line in csv_reader:
            word2value[line[0].strip()] = float(line[1])
        # end for
    # end with

    return word2value
# end def


def test_for_differences(a, b):
    print(len(a), len(b), np.mean(a), np.std(a), np.mean(b), np.std(b))
    print(ranksums(a, b))

# end def


dimension = 'v'
excluded = ['red herring', 'red carpet']
if __name__ == '__main__':
    #word2value = Serialization.load_obj(dimension + '.dict')
    word2value = read_concreteness_scores('original_concreteness.csv')

    full_dist_values = list(word2value.values())

    def_words = read_definition_words('idioms-definitions-final-counts.csv')
    print('total of:', len(def_words), 'in definitions')

    defs_dist_values = list()
    for w in def_words:
        if w not in word2value: continue
        defs_dist_values.append(word2value[w])
    # end for

    full_dist_values = random.sample(full_dist_values, len(defs_dist_values))
    test_for_differences(full_dist_values, defs_dist_values)


# end if
