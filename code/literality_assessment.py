import csv
import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine


def load_embeddings(filename):
    #vectors = np.load(filename_e)
    model = KeyedVectors.load(filename, mmap='r')
    #print('\n'.join(list(model.vocab.keys())))
    return model

# end def


def compute_compositionality(filename, model, stopwords):
    group2lit = dict()
    with open(filename, 'r') as fin, open(filename.replace('.csv', '.score.csv'), 'w') as fout:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)  # skip the header

        csv_writer.writerow(header + ['total', 'min', 'max', 'avg', 'idiom', 'tokens'])

        for line in csv_reader:
            if len(line) == 0: continue
            #line = line + [int(line[3]) + int(line[4])]
            idiom = line[0].strip().lower().replace(' ', '-')
            group = int(line[2])  # a group number of an idiom set
            definition = line[1].split()

            if group in group2lit:
                data = group2lit[group]
                csv_writer.writerow(line + [data[0], data[1], data[2], data[3], data[4]])
                continue
            # end if

            if idiom not in model.vocab:
                print('idiom', idiom, 'not found in embeddings...')
                csv_writer.writerow(line + [-1, -1, -1, -1, 0])
                continue
            # end if

            tokens = list()
            token_embeddings = list()
            for token in line[0].split():
                #if len(token) < 3: continue
                if token in stopwords: continue
                if token not in model.vocab: continue
                token_embeddings.append(model[token])
                tokens.append(token)
            # end for

            sim_avg = 0; sim_min = 0; sim_max = 0; sim_idi = 0
            if len(tokens) > 0:
                sim_min = min([abs(model.similarity(idiom, token)) for token in tokens])
                sim_max = max([abs(model.similarity(idiom, token)) for token in tokens])
                sim_avg = np.mean([abs(model.similarity(idiom, token)) for token in tokens])
                sim_idi = abs(1.0 - cosine(model[idiom], np.average(np.array(token_embeddings), axis=0)))
                group2lit[group] = (sim_min, sim_max, sim_avg, sim_idi, len(tokens))
            # end if

            csv_writer.writerow(line + [sim_min, sim_max, sim_avg, sim_idi, len(tokens)])
        # end for

    # end with

# end def


if __name__ == '__main__':

    model = load_embeddings('wordvectors_both.kv')
    with open('stopwords-short.dat', 'r') as fin: stopwords = fin.read().split()
    filename = 'idioms-definitions-final-counts.csv'  # csv with counts
    compute_compositionality(filename, model, stopwords)


# end if
