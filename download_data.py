from datasets import load_dataset

DATASET_NAME = "tatsu-lab/alpaca"

train_dataset = load_dataset(DATASET_NAME, split="train")
