import pandas as pd
import torch

from constants import BATCH_SIZE, MAX_LEN, MODEL_PATH, RESULTS_DIR, TEST_FILE, TEST_RESULTS_FILE, TOKENIZER_PATH
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from utils import tokenize_data  


# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load preprocessed test data
test_data = pd.read_csv(TEST_FILE, sep='|', header=0, names=['content_processed', 'target'], dtype={'content_processed': str, 'target': int})

# Ensure no missing values and convert content_processed to string
test_data = test_data.fillna('') 
test_data['content_processed'] = test_data['content_processed'].astype(str)

# Tokenizer and model initialization
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)

# Convert test data to Hugging Face dataset format
test_dataset = Dataset.from_dict({'text': test_data['content_processed'].tolist(), 'label': test_data['target'].tolist()})

# Tokenize data using the function from utils.py
test_dataset = tokenize_data(test_dataset, tokenizer, MAX_LEN)

# Evaluate the model
training_args = TrainingArguments(
    output_dir=RESULTS_DIR,  
    per_device_eval_batch_size=BATCH_SIZE,
    no_cuda=not torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
)

# Run evaluation
test_predictions = trainer.predict(test_dataset)
test_preds = test_predictions.predictions.argmax(axis=1)
test_labels = test_data['target'].tolist()

# Evaluate performance
print(f'\nConfusion Matrix:')
print(confusion_matrix(test_labels, test_preds))

print(f'\nClassification Report:')
print(classification_report(test_labels, test_preds))

# Save results to CSV
test_data['test_preds'] = test_preds  
test_data['test_labels'] = test_labels  

# Select only the relevant columns to save
test_results = test_data[['content_processed', 'test_labels', 'test_preds']]

# Save the results as CSV
test_results.to_csv(TEST_RESULTS_FILE, index=False, header=True, sep="|")

print(f'\nTest results saved to {TEST_RESULTS_FILE}')