import torch
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

# Step 2: Load the CB dataset
dataset = load_dataset("super_glue", "cb")

# Step 3: Prepare the XLNet model and tokenizer
model_name = "xlnet-base-cased"
model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = XLNetTokenizer.from_pretrained(model_name)

# # Freeze the last two layers
# for param in model.transformer.layer[-2:].parameters():
#     param.requires_grad = False

# Step 4: Fine-tune the XLNet model using the Trainer API
def preprocess_function(examples):
    # Tokenize input sentence pairs
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length")

train_dataset = dataset["train"].map(preprocess_function, batched=True)
eval_dataset = dataset["validation"].map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

trainer.train()
