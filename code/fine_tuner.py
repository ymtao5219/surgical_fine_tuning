import argparse
from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments, Trainer
import time 

import evaluate
import numpy as np

from data_loader import *

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main(args):
    
    model_name = args.parent_model
    freeze_layers = args.freeze_layers
    task_name = args.task_name

    data_loader = GlueDataloader(task_name, model_name)
    train_dataset, val_dataset = data_loader.get_train_val_split()

    # Model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(train_dataset.features["label"].names))

    def add_prefix(val):
        return "bert.encoder.layer." + str(val)

    # print("layers to freeze", freeze_layers)
    if freeze_layers:
        for i in range(len(freeze_layers)):
            freeze_layers[i] = add_prefix(freeze_layers[i])
        freeze_layers = tuple(freeze_layers)
        print("layers to freeze", freeze_layers)
        for name, param in model.named_parameters():
            if name.startswith(freeze_layers):
                param.requires_grad = False


    # Training arguments
    training_args = TrainingArguments(
        output_dir="checkpoints",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        seed=42,
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Record the start time
    start_time = time.time()

    # Train the model
    trainer.train()

    # Record the end time
    end_time = time.time()

    # Calculate the total training time
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    
    # Save the best model
    trainer.save_model(f"checkpoints/best_model_{args.task_name}")

    # Evaluate the model
    trainer.evaluate()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning a parent model")
    parser.add_argument("--parent_model", type=str, default="bert-base-cased", help="Name of the parent model to use from Hugging Face")
    parser.add_argument("--task_name", type=str, default="wic", help="Name of the task in GLUE/SuperGLUE to fine-tune on")
    parser.add_argument("--freeze_layers", nargs='+', type=int, help="List of which layers to freeze")
    args = parser.parse_args()

    main(args)