from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
import torch

model_name = "sagorsarker/bangla-bert-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4
)

def tokenize(example):

    return tokenizer(
        example["clean_text"],
        padding="max_length",
        truncation=True
    )