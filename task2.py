import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load the dataset (adjust the file path and column names as needed)
df = pd.read_excel('amazon_reviews-1.xlsx')

# Convert ratings to integer and map ratings 1-5 to 0-4
df['overall'] = df['overall'].astype(int)
rating_to_label = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
df['label'] = df['overall'].map(rating_to_label)

# Split the dataset into training (80%) and testing (20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Load the pretrained tokenizer (using BERT in this example)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function
def tokenize_function(example):
    return tokenizer(example['reviewText'], padding='max_length', truncation=True, max_length=256)

# Convert Pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df[['reviewText', 'label']])
test_dataset = Dataset.from_pandas(test_df[['reviewText', 'label']])

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Load a pretrained BERT model for sequence classification with 5 labels
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,               # Use few epochs to avoid overfitting on small datasets
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=100,
    save_strategy="epoch"
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Fine-tune the model
trainer.train()

# Evaluate on the test set
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# Optionally, run final predictions on the test set
predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(-1)
print("Predicted Labels:", predicted_labels)

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_bert')
tokenizer.save_pretrained('./fine_tuned_bert')
