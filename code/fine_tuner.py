import argparse
from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments, Trainer

from data_loader import *

def main(args):
    
    model_name = args.parent_model
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    freeze_layers = args.freeze_layers
    task_name = args.task_name

    data_loader = GlueDataloader(task_name, model_name)
    train_dataset, val_dataset = data_loader.get_train_val_split()

    # Model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(train_dataset.features["label"].names))

    def add_prefix(val):
        return "bert.encoder.layer." + str(val)

    if freeze_layers:
        for i in range(len(freeze_layers)):
            freeze_layers[i] = add_prefix(freeze_layers[i])
        freeze_layers = tuple(freeze_layers)

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
    )

    # Fine-tuning
    trainer.train()

    # Save the best model
    trainer.save_model(f"checkpoints/best_model_{args.attr_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning a parent model")
    parser.add_argument("--parent_model", type=str, default="bert-base-cased", help="Name of the parent model to use from Hugging Face")
    parser.add_argument("--attr_name", type=str, default="sentiment", help="Name of the attribute of interest")
    parser.add_argument("--task_name", type=str, default="cola", help="Name of the dataset to use from Hugging Face")
    parser.add_argument("--freeze_layers", type=str, default=[], help="List of which layers to freeze")
    args = parser.parse_args()

    main(args)