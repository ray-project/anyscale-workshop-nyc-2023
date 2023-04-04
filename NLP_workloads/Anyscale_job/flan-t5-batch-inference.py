from typing import Optional

import anyscale
import ray
import torch
import transformers
from datasets import load_dataset
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.data.preprocessors import BatchMapper
from ray.train.batch_predictor import BatchPredictor
from ray.train.huggingface import HuggingFaceTrainer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import TrainingArguments, Trainer

from predictor import HuggingFaceModelPredictor
from utils import preprocess_function

transformers.set_seed(42)

num_workers = 2
batch_size = 2
use_gpu = True

hf_dataset = load_dataset("tatsu-lab/alpaca", split="train").train_test_split(
    test_size=0.2, seed=57
)

ray_dataset = ray.data.from_huggingface(hf_dataset)
train_ds = ray_dataset["train"].limit(100)
validation_dataset = ray_dataset["test"].limit(100)

batch_preprocessor = BatchMapper(
    preprocess_function, batch_format="pandas", batch_size=4096
)


def trainer_init_per_worker(
    train_dataset: ray.data.Dataset,
    eval_dataset: Optional[ray.data.Dataset] = None,
    **config,
) -> Trainer:
    """
    Initializes a Hugging Face Trainer for training a T5 text generation model.

    Args:
        train_dataset (ray.data.Dataset): The dataset for training the model.
        eval_dataset (ray.data.Dataset, optional): The dataset for evaluating
        the model.
            Defaults to None.
        config: Additional arguments to configure the Trainer.

    Returns:
        Trainer: A Hugging Face Trainer for training the T5 model.
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "google/flan-t5-base"

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    training_args = TrainingArguments(
        "flan-t5-base-finetuned-alpaca",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=config.get("learning_rate", 2e-5),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=config.get("epochs", 4),
        weight_decay=config.get("weight_decay", 0.01),
        push_to_hub=False,
        disable_tqdm=True,
    )

    hf_trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    return hf_trainer


trainer = HuggingFaceTrainer(
    trainer_init_per_worker=trainer_init_per_worker,
    scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
    datasets={
        "train": train_ds,
        "evaluation": validation_dataset,
    },
    run_config=RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="eval_loss",
            checkpoint_score_order="min",
        ),
    ),
    preprocessor=batch_preprocessor,
)

result = trainer.fit()

predictor = BatchPredictor.from_checkpoint(
    checkpoint=result.checkpoint,
    predictor_cls=HuggingFaceModelPredictor,
    model_cls=T5ForConditionalGeneration,
    tokenizer=T5Tokenizer,
    use_gpu=use_gpu,
    device_map="auto",
    torch_dtype=torch.float16,
)

prediction = predictor.predict(
    validation_dataset,
    num_gpus_per_worker=int(use_gpu),
    batch_size=256,
    max_new_tokens=128,
)

input_data_pd = validation_dataset.to_pandas()
prediction_pd = prediction.to_pandas()
input_data_pd.join(prediction_pd, how="inner")

anyscale.job.output({"done": True})
