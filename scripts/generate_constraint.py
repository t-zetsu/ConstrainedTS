import argparse,json,time,flair,torch
from pathlib import Path
from src import RoBERTaScorer
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger
import sys


def read_file(input_file):
    with open(input_file, mode="r") as f:
        lines = [s.replace("\n","") for s in f.readlines()]
    return lines

def read_jsonl(input_file):
    with open(input_file, mode='r', encoding='utf-8') as f:
        lines = [json.loads(s) for s in f.readlines()]
    return lines

def write_jsonl(lines, output_file):
    with open(output_file, mode='w', encoding='utf-8') as f:
        for line in lines:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def vocab_complex(word_dict, grade):
    vocab = []
    for key in word_dict:
        if int(key) > grade:
            vocab += word_dict[key]
    return set(vocab)

def vocab_simple(word_dict, grade):
    vocab = []
    for key in word_dict:
        if int(key) <= grade:
            vocab += word_dict[key]
    return set(vocab)

def word_replacement(complex_word, simple_words, scorer, bs=100, min_score=4):
    candidates = [[complex_word, simple_word] for simple_word in simple_words]
    word_score = [None, min_score]
    for batch in list(chunks(candidates, bs)):
        score = scorer(batch)
        idx = score.index(max(score))
        if score[idx] >= word_score[1]:
            word_score[0] = batch[idx][1]
            word_score[1] = score[idx]
    if word_score[0]:
        return [(complex_word, False), (word_score[0], True)]
    else:
        return [(complex_word, False)]


def gen_constraints(sentences, labels_list, grades, word_dict, scorer):
    all_constraints = []
    flair.device = torch.device('cuda:0') 
    tagger = SequenceTagger.load("flair/pos-english")
    pass_pos = {"JJ","JJR","JJS","NN","NNP","NNPS","NNS","RB","RBR","RBS","VB","VBD","VBG","VBN","VBN","VBP","VBZ"}
    for sent, labels, grade in tqdm(zip(sentences, labels_list, grades), mininterval=1000):
        constraints = []
        pos_dic = {}
        # 難易度辞書
        complex_words = vocab_complex(word_dict, grade) #目標難易度よりも難解な語彙
        simple_words = vocab_simple(word_dict, grade) #目標難易度以下の平易な語彙

        # POS Tagging
        tagged_sent = Sentence(sent)
        tagger.predict(tagged_sent)
        for label in tagged_sent.labels:
            word = label.data_point.text.lower()
            tag = label.data_point.tag
            if word in pos_dic:
                pos_dic[word].add(tag)
            else:
                pos_dic[word] = {tag}

        # Constraint Assembling
        for word, label in zip(sent.split(), labels.split()):
            if word in pos_dic:
                if pos_dic[word] & pass_pos:
                    if label == "DEL" and word in complex_words and [(word, False)] not in constraints:
                        constraints.append([(word, False)])
                    elif label == "KEEP" and word in simple_words and [(word, True)] not in constraints:
                        constraints.append([(word, True)])
                    elif label == "REPL" and word in complex_words:
                        repl_pairs = word_replacement(word, simple_words, scorer)
                        if repl_pairs not in constraints: constraints.append(repl_pairs) # Word Replacement
        all_constraints.append(constraints)

    return all_constraints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='/work/data/datasets/sample.jsonl', help='path to sentences and grade data')
    parser.add_argument("--src_labels", type=str, default='/work/data/datasets/sample.label.txt', help='path to edit operation data')
    parser.add_argument("--dict", type=str, default='/work/data/datasets/Newsela-Auto.train.tf.json', help='path to edit operation data')
    parser.add_argument("--scorer", type=str, default='/work/models/roberta_scorer/')
    parser.add_argument("--output", type=str, default='/work/data/outputs/generations/sample.constraint.jsonl ')
    args = parser.parse_args()

    input_data = read_jsonl(args.input)
    sentences = [line["src_sentence"] for line in input_data]
    grades = [int(line["dst_grade"]) for line in input_data]
    labels_list = read_file(args.src_labels)
    with open(args.dict, 'r') as f:
        word_dict = json.load(f)

    scorer = RoBERTaScorer(Path(args.scorer))
    print("load model",flush=True)

    start = time.time()
    all_constraints = gen_constraints(sentences, labels_list, grades, word_dict, scorer)
    elapsed = time.time() - start
    print (f"constraints: {elapsed} [sec]",flush=True)

    outputs = [{**line, **{"constraint": constraint}} for line, constraint in zip(input_data, all_constraints)]
    write_jsonl(outputs, args.output)

if __name__ == "__main__":
    main()
