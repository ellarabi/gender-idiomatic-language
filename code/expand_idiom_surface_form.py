import csv
import pickle

from pyinflect import getAllInflections


def expand_verbs(idioms, word):
    expanded = list()
    inflections = getAllInflections(word,  pos_type='V')
    verb_forms = set([e[0] for e in inflections.values()])
    for idiom in idioms:
        for form in verb_forms: expanded.append(idiom.replace(word, form))
    # end for

    return expanded
# end def


def expand_personal_pronouns(idiom):
    expanded = list()
    for element in PERSONAL_PRONOUNS: expanded.append(idiom.replace('someone', element))
    return expanded
# end def


def expand_possesive_determiners(idiom):
    expanded = list()
    for element in POSSESIVE_DETERMS: expanded.append(idiom.replace("someone's", element))
    return expanded
# end def


def expand_idiom_forms(filename, word2pos):
    count = 1
    idiomdefs = list(); idiomforms = set()
    with open(filename, 'r') as fin:
        csv_reader = csv.reader(fin,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in csv_reader:
            idiom = line[0].strip()

            expanded = list()
            if "someone's" in idiom: expanded.extend(expand_possesive_determiners(idiom))
            if 'someone' in idiom and "someone's" not in idiom: expanded.extend(expand_personal_pronouns(idiom))
            if len(expanded) == 0: expanded.append(idiom)

            for word in idiom.split():
                if word not in word2pos: continue
                if word2pos[word] == 'VB': expanded.extend(expand_verbs(expanded[:], word))
            # end for

            for item in set(expanded):
                if item in idiomforms:
                    print('existing', item)
                    continue
                idiomdefs.append((item, line[1].strip(), count))
                idiomforms.add(item)
            # end for
            #print(idiom, set(expanded))
            count += 1
        # end for
    # end with

    with open(filename.replace('csv', 'exp.csv'), 'w') as fout:
        csv_writer = csv.writer(fout,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for item in idiomdefs: csv_writer.writerow([item[0], item[1], item[2]])
    # end with

# end def


PERSONAL_PRONOUNS = ['me', 'her', 'him', 'you', 'them']
POSSESIVE_DETERMS = ['my', 'her', 'his', 'your', 'their']

if __name__ == '__main__':
    # using https://pypi.org/project/pyinflect/ for verb tense conversion
    with open('<working dir>/eng.word2pos.pkl', 'rb') as fin:
        word2pos = pickle.load(fin)
    # end with

    # todo: replace one's with someone's in excel
    expand_idiom_forms('idioms-definitions-final.csv', word2pos)

# end if
