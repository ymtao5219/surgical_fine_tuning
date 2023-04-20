import argparse
from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments, Trainer
from datasets import load_dataset


def main(args):
    # Load dataset
    dataset = load_dataset(args.dataset_name)
    

    model_name = args.parent_model
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    # Preprocessing function
    def preprocess_function(examples):
        return tokenizer(examples['sentence'], truncation=True, padding='max_length')

    # Preprocess datasets
    train_dataset = dataset["train"].map(preprocess_function, batched=True)
    val_dataset = dataset["validation"].map(preprocess_function, batched=True)

    # Model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(dataset["train"].features["label"].names))

    # Training arguments
    training_args = TrainingArguments(
        output_dir="checkpoints",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        seed=42,
        save_strategy="best",
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
    parser.add_argument("--parent_model", type=str, default="bert-base-multilingual-cased", help="Name of the parent model to use from Hugging Face")
    parser.add_argument("--attr_name", type=str, default="sentiment", help="Name of the attribute of interest")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to use from Hugging Face")
    
    args = parser.parse_args()

    main(args)
