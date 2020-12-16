import sys
import csv

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import statsmodels.api as sm

from scipy.spatial.distance import jensenshannon


def extract_text_by_gender(filename):
    count = 0
    with open(filename, 'r') as fin, open(filename.replace('csv', 'dat'), 'w') as fout:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in csv_reader:
            fout.write(line[3] + '\n')
            if count % 100000 == 0:
                print(count)
                sys.stdout.flush()
            # end if
            count += 1
        # end for
    # end with

# end def


def extract_text_by_gender_filtered(filename, subreddits):
    allowed = list()
    with open(subreddits, 'r') as fin:
        for line in fin: allowed.append(line.strip())
    # end with

    count = 0
    with open(filename, 'r') as fin, open(filename.replace('csv', 'balanced.dat'), 'w') as fout:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in csv_reader:
            if line[1].strip() not in allowed: continue

            fout.write(line[3] + '\n')
            if count % 100000 == 0:
                print(count)
                sys.stdout.flush()
            # end if
            count += 1
        # end for
    # end with
# end def


def extract_idioms_by_gender(filename, fout_m, fout_f):
    with open(filename, 'r') as fin:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)  # skip the header
        for line in csv_reader:
            idiom = line[0].strip()
            m_count = int(line[3]); f_count = int(line[4])
            fout_m.write('\n'.join(m_count * [idiom.replace(' ', '-')]) + '\n')
            fout_f.write('\n'.join(f_count * [idiom.replace(' ', '-')]) + '\n')
        # end for
    # end with

# end def


def extract_idioms_groups_by_gender(filename, fout_m, fout_f):
    form2count_m = dict()
    form2count_f = dict()
    with open(filename, 'r') as fin:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)  # skip the header
        for line in csv_reader:
            group = int(line[2])
            current = form2count_m.get(group, 0)
            form2count_m[group] = current + int(line[3])
            current = form2count_f.get(group, 0)
            form2count_f[group] = current + int(line[4])
        # end for
    # end with

    for g in form2count_m: fout_m.write('\n'.join(form2count_m[g] * [str(g)]) + '\n')
    for g in form2count_f: fout_f.write('\n'.join(form2count_f[g] * [str(g)]) + '\n')

# end def


def read_logodds_into_dict(filename):
    d = dict()
    with open(filename, 'r') as fin:
        for line in fin: d[line.strip().split()[0]] = float(line.strip().split()[1])
    # end with

    return d
# end def


def test_idioms_language_correlation(filename_i, filename_l):
    d_i = read_logodds_into_dict(filename_i)
    d_l = read_logodds_into_dict(filename_l)

    val_i = list(); val_l = list()
    for word in d_i:
        if word not in d_l: continue
        if -MIN_VAL < d_i[word] < MIN_VAL and -MIN_VAL < d_l[word] < MIN_VAL: continue
        val_i.append(d_i[word]); val_l.append(d_l[word])
    # end for

    assert(len(val_i) == len(val_l)), 'should be of the same length'
    #sim_array = [1 if i*l > 0 else 0 for i, l in zip(val_i, val_l)]
    #overlap = float(sum(sim_array))/len(sim_array)

    print('values arrays length:', len(val_i))
    return pearsonr(val_i, val_l)

# end def


def extract_gender_language_metrics(stopwords):
    word2gender = dict()
    with open('log_odds.gender.language.out', 'r') as fin:
        for line in fin:
            tokens = line.split('\t')
            if len(tokens) < 2: continue
            word2gender[tokens[0]] = float(tokens[1])
        # end for
    # end with
    print('loaded', len(word2gender), 'words...')

    idiom2def = dict()
    filename = '../idioms-definitions-final-counts.csv'
    with open(filename, 'r') as fin:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)  # skip the header
        for line in csv_reader:
            idiom2def[line[0].lower().replace(' ', '-')] = line[1]
        # end for
    # end with
    print('loaded', len(idiom2def), 'idiom definitions...')

    filename = 'log_odds.idioms.out'
    with open(filename, 'r') as fin, open('idioms-gender-metrics.wos.func.csv', 'w') as fout:
        csv_writer = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in fin:
            if len(line) == 0: continue
            idiom = line.split()[0]
            definition = idiom2def[idiom]

            idiom_scores = list()
            for word in idiom.replace('-', ' ').split():
                if word in stopwords: continue
                idiom_scores.append(word2gender.get(word, 0))
            # end for

            definition_scores = list()
            for word in definition.split():
                if word in stopwords: continue
                definition_scores.append(word2gender.get(word, 0))
            # end for

            csv_writer.writerow([idiom, line.split()[1], np.mean(idiom_scores), np.mean(definition_scores)])

        # end for
    # end with
    print('computed idiom and definition scores...')

# end def


def extract_gender_language_metrics_groups(stopwords):
    word2gender = dict()
    with open('log_odds.gender.language.out', 'r') as fin:
        for line in fin:
            tokens = line.split('\t')
            if len(tokens) < 2: continue
            word2gender[tokens[0]] = float(tokens[1])
        # end for
    # end with
    print('loaded', len(word2gender), 'words...')

    group2props = dict()
    filename = '../idioms-definitions-final-counts.csv'
    with open(filename, 'r') as fin:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)  # skip the header
        for line in csv_reader:
            if line[2] in group2props: continue
            group2props[line[2]] = (line[0].lower(), line[1])
        # end for
    # end with
    print('loaded', len(group2props), 'idiom forms and definitions...')

    filename = 'log_odds.idioms.groups.out'
    with open(filename, 'r') as fin, open('idioms-gender-metrics.wos.func.csv', 'w') as fout:
        csv_writer = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in fin:
            if len(line) == 0: continue
            form, definition = group2props[line.split()[0]]

            idiom_scores = list()
            for word in form.replace('-', ' ').split():
                if word in stopwords: continue
                idiom_scores.append(word2gender.get(word, 0))
            # end for

            definition_scores = list()
            for word in definition.split():
                if word in stopwords: continue
                definition_scores.append(word2gender.get(word, 0))
            # end for

            csv_writer.writerow([line.split()[0], line.split()[1], np.mean(idiom_scores),
                np.mean(definition_scores)])

        # end for
    # end with
    print('computed idiom and definition scores...')

# end def


def enrich_with_log_frequency(counts, filename):
    idiom2count = dict(); idiom2form = dict()
    with open(counts, 'r')as fin:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)  # skip the header
        for line in csv_reader:
            group = int(line[2])
            current = idiom2count.get(group, 0)
            idiom2count[group] = current + (int(line[3]) + int(line[4]))
            if group in idiom2form: continue
            idiom2form[group] = line[0]
        # end for
    # end with

    with open(filename, 'r') as fin, open(filename.replace('out', 'counts.csv'), 'w') as fout:
        csv_writer = csv.writer(fout,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in fin:
            if len(line.split()) < 2: continue
            group = int(line.split()[0])
            idiom = idiom2form[group].replace('-', ' ')
            csv_writer.writerow([group, idiom, line.split()[1], idiom2count[group]])
        # end for
    # end with
# end def


def compare_distribution(filename):
    form2count_m = dict()
    form2count_f = dict()
    with open(filename, 'r') as fin:
        csv_reader = csv.reader(fin, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)  # skip the header
        for line in csv_reader:
            group = int(line[2])
            current = form2count_m.get(group, 0)
            form2count_m[group] = current + int(line[3])
            current = form2count_f.get(group, 0)
            form2count_f[group] = current + int(line[4])
        # end for
    # end with

    counts_m = list(); counts_f = list()
    for g in form2count_m:
        counts_m.append(float(form2count_m[g]))
        counts_f.append(float(form2count_f[g]))
    # end for

    prob_m = np.array(counts_m) / sum(counts_m)
    prob_f = np.array(counts_f) / sum(counts_f)
    print(jensenshannon(prob_m, prob_f))

    form2count_m_alt = dict()
    with open('../idioms-definitions-final-counts.m.2.csv', 'r') as fin:
        csv_reader = csv.reader(fin, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)  # skip the header
        for line in csv_reader:
            group = int(line[2])
            current = form2count_m_alt.get(group, 0)
            form2count_m_alt[group] = current + int(line[3])
        # end for
    # end with

    counts_m = list()
    for g in form2count_m_alt:
        counts_m.append(float(form2count_m_alt[g]))
    # end for

    prob_m_alt = np.array(counts_m) / sum(counts_m)
    assert(len(prob_m) == len(prob_m_alt))
    print(jensenshannon(prob_m, prob_m_alt))

# end def


def test_correlations(filename):
    s_score = list(); i_score = list(); d_score = list()
    with open(filename, 'r') as fin:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in csv_reader:
            if line[1] == 'nan' or line[2] == 'nan' or line[3] == 'nan': continue
            if abs(float(line[1])) < MIN_VAL: continue
            i_score.append(float(line[1]))  # idiom gender score
            s_score.append(float(line[2]))  # mean idiom words gender score
            d_score.append(float(line[3]))  # mean idiom definition words gender score
        # end for
    # end with

    print('spearman(i_score, s_score):', spearmanr(i_score, s_score))
    print('spearman(i_score, d_score):', spearmanr(i_score, d_score))
    print('spearman(s_score, d_score):', spearmanr(s_score, d_score))

    regressors = np.column_stack((i_score, d_score))
    regressors = sm.add_constant(regressors)  # add costant
    model = sm.OLS(s_score, regressors).fit()
    #print(model.summary())

    same_sign_form = list()
    same_sign_definition = list()
    for s, i, d in zip(s_score, i_score, d_score):
        same_sign_form.append(1 if s*i > 0 else 0)
        same_sign_definition.append(1 if s * d > 0 else 0)
    # end for

    print('same_sign_form:', sum(same_sign_form), len(same_sign_form))
    print('same_sign_definition:', sum(same_sign_definition), len(same_sign_definition))

# end def


def combine_idiom_with_score():
    group2idiom = dict()
    with open('../idioms-definitions-final-counts.csv', 'r') as fin:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)
        for line in csv_reader: group2idiom[int(line[2])] = line[0]
    # end with

    filename = 'idioms-gender-metrics.wos.func.csv'
    with open(filename, 'r') as fin, open('idioms-gender-score.csv', 'w') as fout:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in csv_reader:
            if int(line[0]) not in group2idiom:
                print('group', line[0], 'not in idioms...')
                continue
            # end if
            outline = [line[0], group2idiom[int(line[0])], line[1]]
            csv_writer.writerow(outline)
        # end for
    # end with

# end def


MIN_VAL = 0
if __name__ == '__main__':
    # todo: extract general gender language
    dirname = '<working dir>/gender-idioms/'
    #extract_text_by_gender(dirname + 'data.clean.F.relationship.csv')
    #extract_text_by_gender(dirname + 'data.clean.M.2.csv')

    #extract_text_by_gender_filtered(dirname + 'data.clean.M.1.csv', '../balanced.40.subreddits')
    #extract_text_by_gender_filtered(dirname + 'data.clean.F.csv', '../balanced.40.subreddits')

    '''
    # todo: extract idiomatic expressions
    fout_m = open('idioms.M.groups.dat', 'w'); fout_f = open('idioms.F.groups.dat', 'w')
    extract_idioms_groups_by_gender('../idioms-definitions-final-counts.csv', fout_m, fout_f)
    #extract_idioms_by_gender('../idioms-definitions-final-counts.csv', fout_m, fout_f)
    fout_m.close(); fout_f.close()
    '''

    '''
    #enrich_with_log_frequency('../idioms-definitions-final-counts.csv', 'log_odds.idioms.groups.out')

    #compare_distribution('../idioms-definitions-final-counts.csv')


    with open('stopwords.s.dat', 'r') as fin: stopwords = fin.read().split()
    extract_gender_language_metrics_groups(stopwords)
    #extract_gender_language_metrics(stopwords)

    print('corr of idiom, surface and def...')
    test_correlations(sys.argv[1])
    '''

    combine_idiom_with_score()


# end if
