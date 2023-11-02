from transformers import AutoTokenizer, AutoModelForTokenClassification
import codecs, torch, transformers
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import argparse
import os
import glob, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def read_jsonl(input_file):
    with open(input_file, mode='r', encoding='utf-8') as f:
        lines = [json.loads(s) for s in f.readlines()]
    return lines

def read_file(input_file):
    with open(input_file, mode='r', encoding='utf-8') as f:
        lines = [s.replace("\n","") for s in f.readlines()]
    return lines

class LabelDataset(torch.utils.data.Dataset):
	def __init__(self, encodings):
		self.encodings = encodings

	def __getitem__(self, idx):
		item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
		return item

	def __len__(self):
		return len(self.encodings["input_ids"])

class LabelEstimator(pl.LightningModule):
	def __init__(self, corpus_path, pretrained_model, label_list, batch_size, learning_rate, n_level):
		super().__init__()
		self.save_hyperparameters()        
		self.corpus_path = corpus_path
		self.label_list = label_list
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
		self.model = AutoModelForTokenClassification.from_pretrained(pretrained_model, num_labels=len(self.label_list)+1)
		# self.model = AutoModelForTokenClassification.from_pretrained(pretrained_model)
		# self.grade_dic = {'additional_special_tokens': ["<TWELVE>","<ELEVEN>","<TEN>","<NINE>","<EIGHT>","<SEVEN>","<SIX>","<FIVE>","<FOUR>","<THREE>","<TWO>"]}
		self.grade_dic = {'additional_special_tokens': [f"<{i}>" for i in range(n_level)]}
		num_added_toks = self.tokenizer.add_special_tokens(self.grade_dic)
		self.model.resize_token_embeddings(len(self.tokenizer))

	def forward(self, x):
		outputs = self.model(**x)
		predictions = np.argmax(outputs.logits.cpu().detach().numpy(), axis=-1)
		predictions = [p.tolist() for p in predictions]
		return predictions
		
	def on_predict_start(self):
		self.predictions = []
	
	def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
		self.predictions += outputs
		
	def on_predict_end(self):
		self.predictions = self.realign_labels(self.predictions)
		
	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
		return optimizer

	def prepare_data(self):
		self.data = {}
		self.data["sample"] = {
			"tokens": [[f"<{line['dst_grade']}>"] + line["src_sentence"].split() for line in read_jsonl(self.input_file)],
			}
		# self.data["sample"] = {
		# 	"tokens": [line.split() for line in read_file(self.input_file)],
		# 	}

	def predict_dataloader(self):
		inputs = self.tokenizer(self.data["sample"]["tokens"], is_split_into_words=True, padding=True, return_tensors="pt", max_length=512, truncation=True)
		# inputs = self.tokenize_and_align_labels(self.data["sample"])
		self.word_ids = [inputs.word_ids(idx) for idx in range(len(inputs["input_ids"]))]
		return DataLoader(LabelDataset(inputs), batch_size=self.batch_size, num_workers=16, shuffle=False)

	def realign_labels(self, pred_labels):
		preds = []
		for i, pred in enumerate(pred_labels):
			word_ids = self.word_ids[i]
			msk = []
			for idx in range(len(word_ids)):
				if word_ids[idx] == None:
					msk.append(False)
				else:
					if word_ids[idx] == word_ids[idx-1]:
						msk.append(False)   
					else:
						msk.append(True)
			preds.append([self.label_list[pred[j]] for j in range(len(msk)) if msk[j]])
		return preds

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_file')
	parser.add_argument('--output_file')
	parser.add_argument('--model_dir')
	parser.add_argument('--seed', default=42)
	args = parser.parse_args()

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	gpus = torch.cuda.device_count()

	print(args.model_dir)
	# model_path = "/work/models/edit_label_estimator_Newsela-Auto"
	estimator = LabelEstimator.load_from_checkpoint(
		# checkpoint_path=glob.glob(f'{args.model_dir}/checkpoints/*.ckpt')[0],
		checkpoint_path=glob.glob(f'{args.model_dir}/checkpoints/*.ckpt')[0],
		hparams_file=f'{args.model_dir}/hparams.yaml',
		map_location=None
	)

	estimator.input_file = args.input_file

	trainer = pl.Trainer(
		accelerator='gpu',
		devices=gpus
	)

	trainer.predict(estimator)

	predictions = [" ".join(preds[1:]) for preds in estimator.predictions]              
	with open(args.output_file, "w", encoding="utf-8") as f:
		f.write("\n".join(predictions))
	
if __name__ == "__main__":
	main()