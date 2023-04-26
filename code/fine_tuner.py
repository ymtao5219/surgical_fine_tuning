import argparse
from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments, Trainer
from datasets import load_dataset


def main(args):
    # Load dataset
    dataset = load_dataset(args.dataset_name)
    
    model_name = args.parent_model
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    freeze_layers = args.freeze_layers
    dataset_name = args.dataset_name

    if dataset_name in ["mnli_mismatched", "mnli_matched"]:
        dataset_train_split = load_dataset('glue', 'mnli', split='train')
        dataset_val_split = load_dataset('glue', dataset_name, split='validation')
    else:
        dataset_train_split = load_dataset('glue', dataset_name, split='train')
        dataset_val_split = load_dataset('glue', dataset_name, split='validation')
    
    train_dataset, val_dataset = dataset_train_split, dataset_val_split
    # Preprocessing function
  
    if dataset_name in ["mrpc", "stsb", "rte", "wnli"]:
        def preprocess_function(examples):
            return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')
    elif dataset_name in ["qqp"]:
        def preprocess_function(examples):
            return tokenizer(examples['question1'], examples['question2'], truncation=True, padding='max_length')
    elif dataset_name in ["mnli_mismatched", "mnli_matched", "ax"]:
        def preprocess_function(examples):
            return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length')
    elif dataset_name in ["qnli"]:
        def preprocess_function(examples):
            return tokenizer(examples['question'], examples['sentence'], truncation=True, padding='max_length')
    else:
        def preprocess_function(examples):
            return tokenizer(examples['sentence'], truncation=True, padding='max_length')
    
    # Preprocess datasets
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)

    # Model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(dataset_train_split.features["label"].names))


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
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
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
    parser.add_argument("--parent_model", type=str, default="bert-base-multilingual-cased", help="Name of the parent model to use from Hugging Face")
    parser.add_argument("--attr_name", type=str, default="sentiment", help="Name of the attribute of interest")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to use from Hugging Face")
    parser.add_argument("--freeze_layers", type=str, default=[], help="List of which layers to freeze")
    args = parser.parse_args()

    main(args)
