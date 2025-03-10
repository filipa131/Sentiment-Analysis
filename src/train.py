import os
import pandas as pd
import torch

from constants import BATCH_SIZE, EPOCHS, EVALUATION_DIR, LEARNING_RATE, MAX_LEN, MODEL_DIR, MODEL_PATH, N_SPLITS, PROCESSED_DATA_FILE, RESULTS_DIR, TEST_FILE, TOKENIZER_PATH
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from utils import add_prompt, create_dataset_from_splits, tokenize_data


# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load preprocessed data
data = pd.read_csv(PROCESSED_DATA_FILE, sep='|', header=None, names=['content', 'content_processed', 'target'], usecols=['content_processed', 'target'], skiprows=1)
data = data.astype('str')
data['target'] = data['target'].astype(int)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(data['content_processed'], data['target'], stratify=data['target'], test_size=0.3, random_state=42)

test_data = pd.DataFrame({
    'content_processed': X_test,
    'target': y_test
})

# Save test data
os.makedirs(EVALUATION_DIR, exist_ok=True)
test_data.to_csv(TEST_FILE, index=False, header=True, sep="|")

# Make directories for saving results
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Tokenizer Initialization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

best_model = None
best_f1_score = 0.0  

for lr in LEARNING_RATE:
    print(f"Training with learning rate: {lr}")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"Training fold {fold + 1}/{N_SPLITS}...")
        
        # Split data for fold
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Add a prompt for sentiment analysis
        train_data, val_data = create_dataset_from_splits(X_train_fold.apply(add_prompt), X_val_fold.apply(add_prompt), y_train_fold, y_val_fold)

        # Tokenize data
        train_data = tokenize_data(train_data, tokenizer, MAX_LEN)
        val_data = tokenize_data(val_data, tokenizer, MAX_LEN)

        # Load model
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        model.to(device)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(RESULTS_DIR, f'results_fold{fold}_lr{lr}'),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            learning_rate=lr, 
        )

        # Define trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
        )

        # Train model
        trainer.train()
        
        # Evaluate the model on the validation dataset
        val_predictions = trainer.predict(val_data)
        val_preds = val_predictions.predictions.argmax(axis=1)
        val_labels = val_data['label']

        # Evaluate performance
        val_report = classification_report(val_labels, val_preds, output_dict=True)

        print(f'\nConfusion matrix (BERT) - Fold {fold + 1}:')
        print(confusion_matrix(val_labels, val_preds))
        print(f'\nClassification report (BERT) - Fold {fold + 1}:')
        print(val_report)

        # Extract F1-score from the classification report
        f1_score = val_report['weighted avg']['f1-score']

        # Check if this model is the best performing one based on F1-score
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_model = model  # Save the best model


# Save the best model after all folds and hyperparameter sets
if best_model is not None:
    best_model.save_pretrained(MODEL_PATH)  
    tokenizer.save_pretrained(TOKENIZER_PATH)
    print(f"The best performing model has been saved at {MODEL_PATH}")