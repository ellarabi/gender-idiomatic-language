from gensim.models import Word2Vec
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_f", default="../data/text_f_final.txt", help="input filename (female)")
parser.add_argument("--input_m", default="../data/text_m_final.txt", help="input filename (male)")
parser.add_argument("--vectors_f", default="../data/wordvectors_f_final_min50.kv", help="vector file (female)")
parser.add_argument("--vectors_m", default="../data/wordvectors_m_final_min50.kv", help="vector file (male)")



def extract_sentences(filename):
    sentences = []
    with open(filename) as f:
        for l in tqdm(f):
            l = l.lower()
            sentences.append(l.split())
    return sentences


if __name__ == '__main__':

    args = parser.parse_args()

    # traing embeddings with M data
    print("extracting input for gensim, M...")
    sentences_m = extract_sentences(args.input_m)
    print("training M...")
    model_m = Word2Vec(sentences_m, size=300, window=5, min_count=50, workers=8)
    model_m.wv.save(args.vectors_m)

    # traing embeddings with F data
    print("extracting input for gensim, F...")
    sentences_f = extract_sentences(args.input_f)
    print("training F...")
    model_f = Word2Vec(sentences_f, size=300, window=5, min_count=50, workers=8)
    model_f.wv.save(args.vectors_f)



