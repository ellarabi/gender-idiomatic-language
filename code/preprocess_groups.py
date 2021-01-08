import sys
import argparse
import random
import numpy as np
import csv
from tqdm import tqdm
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

parser = argparse.ArgumentParser()
parser.add_argument("--input_f", default="../data/data.clean.F.csv", help="input filename (female)")
parser.add_argument("--input_m", default="../data/data.clean.M.csv", help="input filename (male)")
parser.add_argument("--preprocessed_f", default="../data/text_f_final.txt", help="preprocessed filename (female)")
parser.add_argument("--preprocessed_m", default="../data/text_m_final.txt", help="preprocessed filename (male)")
parser.add_argument("--idioms", default="../data/idioms-definitions-groups-for-embeddings-v2.csv", help="idioms filename")


def extract_idiom_groups(filename):
    idiom_groups = {}
    all_idioms = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if row[0]!= "idiom":
                idiom = row[0]

                # lower case and remove hyphens and commas
                idiom = idiom.lower().replace("-", " ")
                idiom = idiom.replace(",", "")

                # group idioms that belong together
                group_num = row[2]
                if group_num not in idiom_groups:
                    idiom_groups[group_num] = []
                idiom_groups[group_num].append(idiom)
                all_idioms.append(idiom)

    return idiom_groups, all_idioms


def change_raw_text(filename, preprocessed, all_idioms, idioms_single, nlp):

    with open(filename) as f, open(preprocessed, "w") as f_out:
        reader = csv.reader(f)
        for row in tqdm(reader):

            l = row[3].lower()
            l = l.replace("-", " ")

            # find idioms and change them to single tokens
            for idiom in all_idioms:
                l = l.replace(idiom, idioms_single[idiom])

            # tokenize
            doc = nlp(l)
            l = " ".join([token.text for token in doc])

            f_out.write(l + "\n")


def main():

    args = parser.parse_args()
    nlp = English()

    # extract idioms split into groups
    idiom_groups, all_idioms = extract_idiom_groups(args.idioms)

    # find a "generic" idiom to represent each group
    idioms_single = {}
    for num in idiom_groups:
        generic = "".join(idiom_groups[num][0].split())
        for idiom in idiom_groups[num]:
            idioms_single[idiom] = generic

    # normalize data, and replace idioms with a single-token that represents them
    change_raw_text(args.input_f, args.preprocessed_f, all_idioms, idioms_single, nlp)
    change_raw_text(args.input_m, args.preprocessed_m, all_idioms, idioms_single, nlp)


if __name__ == '__main__':

    main()


