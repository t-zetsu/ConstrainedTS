from pathlib import Path
from typing import Optional
from typing import List
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer


SEED = 42


class RoBERTaScorer():
    """Paraphrase scorer based on fine-tuned RoBERTa."""

    def __init__(self, model_dir: Path,
                 tokenizer_name: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        model_dir : Path
            Path to directory containing fine-tuned model and h-params file.
        tokenizer_name : str, optional
            Tokenizer name, by default the same name as one of model in
            `model_dir`.
        """
        super().__init__()

        pl.seed_everything(SEED, workers=True)
        f_ckpt = str(next(model_dir.glob('**/epoch=*-step=*.ckpt')))
        f_params = str(model_dir / 'hparams.yaml')

        self.model = RoBERTaModel.load_from_checkpoint(f_ckpt,
                                                       hparams_file=f_params)
        self.model.eval()
        tokenizer_name = tokenizer_name if tokenizer_name is not None \
            else self.model.hparams['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.gpu = torch.cuda.is_available()
        if self.gpu:
            self.model.cuda()

    def __call__(self, batch: List[List]) -> List[float]:
        return self.scoring_batch(batch)

    def scoring(self, src: str, tgt: str) -> float:
        """
        Score pairwise word/phrase (`src` and `tgt`).

        Parameters
        ----------
        src, tgt : str
            Word/phrase pair whose similarity will be estimated.

        Returns
        -------
        float
            Score (or similarity) between `src` and `tgt` by RoBERTa.
        """
        encodings = self.tokenizer(
            [[src, tgt]], return_tensors='pt', padding=True
        )
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        if self.gpu:
            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')

        y_hat = self.model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=None)
        score = y_hat.squeeze(1).to('cpu').detach().numpy().copy()
        return score[0]

    def scoring_batch(self, batch: List[List]) -> List[float]:
        """
        Score word/phrase pairs using mini-batch.

        Parameters
        ----------
        batch : List[List]
            Batch e.g. `[['src1', 'tgt1'], ['src2', 'tgt2'], ...]`.

        Returns
        -------
        List[float]
            Scores of word/phrase pairs in batch e.g. `[0.45, 0.32, ...]`.
        """
        encodings = self.tokenizer(batch, return_tensors='pt', padding=True)
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        if self.gpu:
            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
        y_hat = self.model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=None)
        scores = y_hat.squeeze(1).to('cpu').detach().numpy().copy()
        return scores.tolist()


class RoBERTaModel(pl.LightningModule):
    """Fine-tuner xBERT for regression toward PPDB human-labeled scores."""

    def __init__(self, model_name: str = 'roberta-large',
                 lr: float = 2e-5, total_step: int = None):
        """
        Initialize RegressionModel instance.

        Parameters
        ----------
        model_name : str, optional
            Pre-trained model name to fine-tune, by default 'roberta-large'.
        lr : float, optional
            Max learning rate for training using linear schedule with warmup,
            by default 5e-5.
        total_step : int, optional
            Total step size during traning, by default None.
        """
        super(RoBERTaModel, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(self.roberta.config.hidden_dropout_prob)
        self.W = nn.Linear(self.roberta.config.hidden_size, 1)
        self.lr = lr
        self.total_step = total_step
        self.save_hyperparameters('model_name', 'lr')

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outs = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        cls_stacked = torch.stack([outs[0][i][0] for i in range(len(outs[0]))])
        cls_stacked = self.dropout(cls_stacked)
        pred = self.W(cls_stacked)
        return pred