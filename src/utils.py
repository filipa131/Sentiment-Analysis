import gzip
import re
import torch

from constants import MAX_LEN, MODEL_PATH, TOKENIZER_PATH
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification


def parse_line(line: str):
    """Function for parsing line in csv file."""
    match = re.match(r"(\S+ \S+) INFO\s+\d+ MSG \[msgid:(\d+)\] \[sender:(.*?)\] \[flag:(.*?)\] \[target:(\d+)\] \[content:'(.*)'\]", line)
    if match:
        return match.groups()
    return None

def read_and_parse_file(file_path: str):
    """Reads and parses the raw data from a .csv.gz file."""
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8', errors='replace') as file:
        for line in file:  
            line = line.strip()
            parsed = parse_line(line)
            if parsed:
                data.append(parsed)
    return data


def add_prompt(text: str) -> str:
    """Adds a prompt for sentiment analysis task."""
    return "Sentiment: " + text

def tokenize_data(data, tokenizer, max_len: int):
    """Tokenizes the data."""
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_len)

    return data.map(tokenize_function, batched=True)
    
def create_dataset_from_splits(train_data, val_data, train_labels, val_labels):
    """Converts splits of data into Hugging Face dataset format."""
    train_dataset = Dataset.from_dict({'text': train_data.tolist(), 'label': train_labels.tolist()})
    val_dataset = Dataset.from_dict({'text': val_data.tolist(), 'label': val_labels.tolist()})
    return train_dataset, val_dataset


def load_model_and_tokenizer():
    """Loads the model and tokenizer once for efficient inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    return tokenizer, model, device

def predict_message(message: str, tokenizer, model, device) -> str:
    """Predicts the sentiment of a single message using a preloaded model and tokenizer."""
    inputs = tokenizer(
        message, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return "Positive" if prediction == 0 else "Negative"