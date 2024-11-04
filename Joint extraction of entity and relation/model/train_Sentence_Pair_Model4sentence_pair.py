import os
os.environ['NCCL_P2P_LEVEL'] = 'NVL'
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['NCCL_DEBUG'] = 'INFO'

import logging
import random
from dataclasses import dataclass, field
from typing import Optional
from easydict import EasyDict

from torch.utils.data import DataLoader
from transformers import (get_linear_schedule_with_warmup,
                          AutoTokenizer,
                          TrainingArguments,
                          HfArgumentParser,
                          Trainer,
                          set_seed)

from util import json_file_load
from nn_utils.trainer import SentencePairClassificationTrainer
from bm_gbd_dataset import SentencePairDataset,SentnecePairCollateFn
from model.sentence_pair_classification import BERTGLCN

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    ),
    model_save_path:Optional[str] = field(
        default='./SentencePairModel'
    ),
    device:Optional[str] = field(
        default="auto",
        metadata={
            "choices": ["auto", "cuda", "cpu"],
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text/json file)."})
    eval_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text/json file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."}
    )
    label_config_path:Optional[str] = field(default=None)
    max_seq_length: Optional[int] = field(default=512)

@dataclass
class SentencePairTrainingArguments(TrainingArguments):
    dropout: Optional[float] = field(default=0.1)
    modules_to_save: Optional[str] = field(default=None)

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,SentencePairTrainingArguments))
    model_args,dataset_args,training_args= parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)


    label2id = json_file_load(dataset_args.label_config_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # Create loaders for datasets

    dataset_args.train_file = os.path.join(dataset_args.dataset_path,dataset_args.train_file)
    dataset_args.eval_file = os.path.join(dataset_args.dataset_path,dataset_args.eval_file)
    dataset_args.validation_file = os.path.join(dataset_args.dataset_path,dataset_args.validation_file)
    
     
    training_dataset = SentencePairDataset(dataset_path=dataset_args.train_file,
                              tokenizer=tokenizer,
                              labels2id=label2id,
                              max_len_seq=dataset_args.max_seq_length,
                              device = model_args.device)
    val_dataset = SentencePairDataset(dataset_path=dataset_args.validation_file,
                     tokenizer=tokenizer,
                     labels2id=label2id,
                     max_len_seq=dataset_args.max_seq_length,
                     device = model_args.device)

    # training_dataloader = DataLoader(
    #     dataset = training_dataset,
    #     batch_size = training_args.per_device_train_batch_size,
    #     collate_fn=SentnecePairCollateFn(),
    #     shuffle=True
    # )
    # val_dataloader = DataLoader(
    #     dataset = val_dataset,
    #     batch_size = training_args.per_device_train_batch_size,
    #     collate_fn=SentnecePairCollateFn(),
    #     shuffle=True
    # )

    sentence_pair_mdoel = BERTGLCN(
        pretrained_model_name_or_path = model_args.model_name_or_path,
        dropout_prob=training_args.dropout,
        num_labels=len(label2id)*2 - 1,
        id2label={str(v): k for k, v in label2id.items()}
    )
    
    trainer = Trainer(
        model= sentence_pair_mdoel,
        tokenizer = tokenizer,
        train_dataset=training_dataset,
        eval_dataset=val_dataset,
        data_collator = SentnecePairCollateFn(num_labels=len(label2id)),
        args = training_args
    )
    trainer.train()

if __name__=="__main__":
    main()
