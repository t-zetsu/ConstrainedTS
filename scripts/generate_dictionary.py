import argparse,re,sys,json
from collections import Counter
from tqdm import tqdm

def read_jsonl(input_file):
    return [json.loads(x) for x in open(input_file, mode='r', encoding='utf-8').readlines()]

def generate_tf_dictionary(sents, grades, rm_symbol, rm_digit):
    tf_dict = {}

    sents = [sent.lower() for sent in sents]
    if rm_symbol:
        sents = [re.sub(re.compile("[!-/:-@[-`{-~]"), '', sent) for sent in sents]
    if rm_digit:
        sents = [re.sub(re.compile("[0-9]"), '', sent) for sent in sents]

    print('Counting words ...')
    for grade, sent in tqdm(zip(grades, sents)):
        if grade not in tf_dict:
            tf_dict[grade] = Counter()
        tf_dict[grade] += Counter(sent.split())

    print('Calculating Term Frequency ...')
    for grade, words_counter in tf_dict.items(): 
        words_dict = dict(words_counter)
        words_sum = sum(words_dict.values())
        for word, word_num in words_dict.items():
            words_dict[word] = (word_num/words_sum)
        tf_dict[grade] = dict(sorted(words_dict.items(),key=lambda x:x[1],reverse=True))        
    return tf_dict

def generate_grades_dictionary(tf_dict):
    grades_dict = {}

    words_all = set()
    for grade in list(tf_dict.keys()):
        words_all |= set(tf_dict[grade].keys())

    print('Searching grades of words ...')
    for word in words_all:
        max_grade, max_value = 0, 0
        for grade in list(tf_dict.keys()):
            if word in tf_dict[grade]:
                if tf_dict[grade][word] > max_value:
                    max_grade, max_value = grade, tf_dict[grade][word]
        if max_grade in grades_dict:
            grades_dict[max_grade].append(word)
        else:
            grades_dict[max_grade] = [word]
            
    return grades_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',help='path to sentences and grade data')
    parser.add_argument('--output',help='path to output file')
    parser.add_argument('--rm_symbol', action='store_true',
                        help='remove symbol from sentences')
    parser.add_argument('--rm_digit', action='store_true',
                        help='remove digit from sentences')
    args = parser.parse_args()

    input_data = read_jsonl(args.input)
    sentences = [line["src_sentence"] for line in input_data] + [line["dst_sentence"] for line in input_data]
    grades = [line["src_grade"] for line in input_data] + [line["dst_grade"] for line in input_data]
    tf_dict = generate_tf_dictionary(sentences, grades, args.rm_symbol, args.rm_digit)

    grades_dict = generate_grades_dictionary(tf_dict)

    with open(args.output, 'w') as f:
        json.dump(grades_dict, f)

if __name__ == "__main__":
    main()