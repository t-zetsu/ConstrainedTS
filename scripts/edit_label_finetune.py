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
import json
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

		# self.validation_step_outputs = []
		# self.test_step_outputs = []

	def forward(self, x):
		outputs = self.model(**x)
		predictions = np.argmax(outputs.logits.cpu().detach().numpy(), axis=-1)
		predictions = [p.tolist() for p in predictions]
		return predictions
	
	def step(self, batch, batch_idx):
		# training_step defined the train loop.
		# It is independent of forward
		outputs = self.model(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
		loss = outputs.loss

		logs = {"loss": loss}
		return loss, logs

	def _shared_eval_step(self, batch, batch_idx):
		outputs = self.model(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
		loss = outputs.loss
		predictions = np.argmax(outputs.logits.cpu().detach(), axis=-1)
		truth_labels = batch['labels'].cpu().detach().numpy()
		truth_and_pred = {'truth': truth_labels, 'pred': predictions}
		return loss, truth_and_pred

	def evaluation(self, outputs):
		truth_labels = []
		pred_labels = []
		for output in outputs:
			truth_labels += output['truth'].tolist()
			pred_labels += output['pred'].tolist()
		truth_labels = [l[0] for l in truth_labels]
		m = MultiLabelBinarizer().fit(truth_labels+pred_labels)
		eval_score = f1_score(m.transform(truth_labels), m.transform(pred_labels), average='macro')
		logs = {"score": eval_score}
		return logs

	def training_step(self, batch, batch_idx):
		loss, logs = self.step(batch, batch_idx)
		self.log_dict({f"train_{k}": v for k, v in logs.items()})
		return loss

	def validation_step(self, batch, batch_idx):
		loss, predictions = self._shared_eval_step(batch, batch_idx)
		self.log_dict({"val_loss": loss})
		# self.validation_step_outputs.append(loss)
		return predictions
	
	def validation_epoch_end(self, outputs):
		# epoch_average = torch.stack(self.validation_step_outputs).mean()
		# print(epoch_average)
		# logs = self.evaluation(epoch_average)
		logs = self.evaluation(outputs)
		self.log_dict({f"val_{k}": v for k, v in logs.items()})
		# self.validation_step_outputs.clear()

	def test_step(self, batch, batch_idx):
		loss,  predictions = self._shared_eval_step(batch, batch_idx)
		self.log_dict({"test_loss": loss})
		# self.test_step_outputs.append(loss)
		return  predictions

	def test_epoch_end(self, outputs):
		# epoch_average = torch.stack(self.test_step_outputs).mean()
		# logs = self.evaluation(epoch_average)
		logs = self.evaluation(outputs)
		self.log_dict({f"test_{k}": v for k, v in logs.items()})
		# self.test_step_outputs.clear()  # free memory
		
	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
		return optimizer

	def prepare_data(self):
		self.data = {}
		splits = ["train", "valid", "test"]
		for split in splits:
			self.data[split] = {
				"tokens": [[f"<{line['dst_grade']}>"] + line["src_sentence"].split() for line in read_jsonl(f"{self.corpus_path}/{split}.jsonl")],
				"labels": [[self.label_list.index("DEL")] + [self.label_list.index(label) for label in line.split()] for line in read_file(f"{self.corpus_path}/{split}/{split}.label.oracle.txt")]
				} # special token -> DEL

	def train_dataloader(self):
		inputs = self.tokenize_and_align_labels(self.data["train"])
		return DataLoader(LabelDataset(inputs), batch_size=self.batch_size, num_workers=16, shuffle=True)

	def val_dataloader(self):
		inputs = self.tokenize_and_align_labels(self.data["valid"])
		return DataLoader(LabelDataset(inputs), batch_size=self.batch_size, num_workers=16, shuffle=False)

	def test_dataloader(self):
		inputs = self.tokenize_and_align_labels(self.data["test"])
		return DataLoader(LabelDataset(inputs), batch_size=self.batch_size, num_workers=16, shuffle=False)

	def tokenize_and_align_labels(self, examples):
		tokenized_inputs = self.tokenizer(examples["tokens"], is_split_into_words=True, padding=True, return_tensors="pt", max_length=512, truncation=True)
		labels = []
		for i, label in enumerate(examples["labels"]):
			word_ids = tokenized_inputs.word_ids(batch_index=i)
			previous_word_idx = None
			label_ids = []
			for word_idx in word_ids:  # Set the special tokens to -100.
				if word_idx is None:
					label_ids.append(-100)
				elif word_idx != previous_word_idx:  # Only label the first token of a given word.
					label_ids.append(label[word_idx])
				else:
					label_ids.append(-100)
					# label_ids.append(label[word_idx])
				previous_word_idx = word_idx
			labels.append(label_ids)
		tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.long).unsqueeze(1)
		return tokenized_inputs

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
	parser.add_argument('--input_dir')
	parser.add_argument('--model_dir')
	parser.add_argument('--pretrained', default='bert-base-uncased')
	parser.add_argument('--batch_size', default=40)
	parser.add_argument('--initial_lr', default=1e-5)
	parser.add_argument('--seed', default=42)
	parser.add_argument('--n_level', default=2)
	args = parser.parse_args()

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	gpus = torch.cuda.device_count()
	label_list = ["REPL", "KEEP", "DEL"]

	logger = TensorBoardLogger(save_dir=args.model_dir, name='edit_label_estimator_' + args.input_dir.split("/")[-1])
	early_stop_callback = EarlyStopping(
		monitor='val_score',
		min_delta=1e-4,
		patience=5,
		verbose=False,
		mode='max'
	)

	checkpoint_callback = ModelCheckpoint(
		monitor="val_score",
		filename="model-{epoch:02d}-{val_score:.3f}",
		# filepath="model-{epoch:02d}-{val_score:.3f}",
		save_top_k=1,
		mode="max",
	)

	estimator = LabelEstimator(args.input_dir, args.pretrained, label_list, args.batch_size, args.initial_lr, args.n_level)
	trainer = pl.Trainer(
		accelerator='gpu',
		devices=gpus,
		logger=logger,
		callbacks=[checkpoint_callback, early_stop_callback],
		max_epochs=20,
		auto_lr_find=True
	)

	trainer.fit(estimator)
	trainer.test(estimator)
	
if __name__ == "__main__":
	main()