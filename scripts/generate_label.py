import argparse
import json
from tqdm import tqdm

def read_jsonl(input_file):
    return [json.loads(x) for x in open(input_file, mode='r', encoding='utf-8').readlines()]

def read_file(input_file):
	with open(input_file, mode="r") as f:
		lines = [s.replace("\n","") for s in f.readlines()]
	return lines

def generate(src_lines, dst_lines, align_lines):
	label_lines = []
	i=0
	for src, dst, alignment in zip(src_lines, dst_lines, align_lines):
		src_words = src.split()
		dst_words = dst.split()
		label = ["DEL"] * len(src_words)
		for idx in alignment.split():
			src_idx = int(idx.split('-')[0])
			dst_idx = int(idx.split('-')[1])
			if src_idx >= len(src_words) or dst_idx >= len(dst_words):
				print(i)
				continue
			if src_words[src_idx].lower() == dst_words[dst_idx].lower():
				label[src_idx] = "KEEP"
			else:
				label[src_idx] = "REPL"
		label_lines.append(" ".join(label))
		i+=1
	return label_lines

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str)
	args = parser.parse_args()

	splits = ["train", "valid", "test"]
	for split in splits:
		data = read_jsonl(f"{args.input_dir}/{split}.jsonl")
		src_lines = [x["src_sentence"] for x in data]
		dst_lines = [x["dst_sentence"] for x in data]
		align_lines = read_file(f"{args.input_dir}/{split}/{split}.align")
		label_lines = generate(src_lines, dst_lines, align_lines)
		with open(f'{args.input_dir}/{split}/{split}.label.oracle.txt','w') as f:
			f.write('\n'.join(label_lines))

if __name__ == "__main__":
	main()