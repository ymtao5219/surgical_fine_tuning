import torch
from torch.utils.data import DataLoader
from transformers import XLNetForSequenceClassification, XLNetTokenizer, AdamW
from transformers import glue_compute_metrics, glue_convert_examples_to_features, GlueDataset
from transformers import get_linear_schedule_with_warmup

# Step 1: Load the CB dataset
# Download and preprocess the CB dataset into a suitable format

# Step 2: Prepare the XLNet model and tokenizer
model_name = "xlnet-base-cased"
model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = XLNetTokenizer.from_pretrained(model_name)

# Step 3: Fine-tune the XLNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define training hyperparameters
epochs = 3
learning_rate = 2e-5
batch_size = 32

# Create data loader for training
train_dataset = GlueDataset("drive/My Drive/Sem4/696DS/Data/CB/train.json", tokenizer=tokenizer, task="cb", max_length=128)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_data_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Fine-tuning loop
model.train()
for epoch in range(epochs):
    for batch in train_data_loader:
        inputs = tokenizer(batch["sentence"], padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        optimizer.step()
        scheduler.step()

# Step 4: Evaluate the fine-tuned model
# Create data loader for evaluation
eval_dataset = GlueDataset("drive/My Drive/Sem4/696DS/Data/CB/val.json", tokenizer=tokenizer, task="cb", max_length=128)
eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

model.eval()
total_eval_accuracy = 0
for batch in eval_data_loader:
    inputs = tokenizer(batch["sentence"], padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = batch["label"].to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)
    accuracy = torch.sum(predictions == labels).item()
    total_eval_accuracy += accuracy

avg_eval_accuracy = total_eval_accuracy / len(eval_dataset)
print("Evaluation Accuracy:", avg_eval_accuracy)
