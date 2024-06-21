from datasets import load_dataset
from flant5 import CNNDailyMailDataset, Collater
from transformers import T5Tokenizer

def load_and_preprocess_data(teacher_model_name):
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    train_dataset = CNNDailyMailDataset(dataset, split='train')
    tokenizer = T5Tokenizer.from_pretrained(teacher_model_name)
    collator = Collater(tokenizer)
    return train_dataset, tokenizer, collator