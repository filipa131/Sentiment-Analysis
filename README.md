# Sentiment Classification with BERT

This repository contains a BERT-based sentiment analysis pipeline. The model classifies text as either **positive** (0) or **negative** (1). The project includes data preprocessing, model training, evaluation, and inference.

## 📁 Repository Structure
```
sentiment_classification/
│── data/
│   ├── raw/              # Raw, unprocessed data
│   ├── stopwords/        # Stopwords file
│   ├── processed/        # (Generated) Preprocessed data after running preprocess.py
│   ├── evaluation/       # (Generated) Contains test.txt after running train.py
│
│── notebooks/
│   ├── data_exploration.ipynb  # Jupyter Notebook for data exploration
│
│── src/
│   ├── cleaner.py        # Text preprocessing and cleaning functions
│   ├── constants.py      # Global paths, hyperparameters, and configurations
│   ├── evaluate.py       # Evaluates trained model performance
│   ├── inference.py      # Interactive sentiment prediction script
│   ├── preprocess.py     # Preprocesses raw data and generates processed data
│   ├── train.py          # Trains the BERT model
│   ├── utils.py          # Utility functions (e.g., tokenization, dataset parsing)
│
│── model/                # (Generated) Stores trained model files
│── results/              # (Generated) Stores training results and logs
│── .gitignore            # Git ignore file
│── requirements.txt      # Python dependencies
│── setup.py              # Installation script
│── README.md             # Project documentation
```

## 🛠 Setup Instructions

### 1️⃣ Install Dependencies

Clone the repository and install required packages:

```bash
git clone https://github.com/filipa131/sentiment_classification.git
cd sentiment_classification
pip install -r requirements.txt
```

### 2️⃣ Data Preparation

Ensure that your raw dataset is placed inside `data/raw/`. The dataset should follow the example format:

```
2009-30-05 06:57:04 INFO     77525 MSG [msgid:1971371956] [sender:elltotheoh] [flag:NO_QUERY] [target:1] [content:'i need a desk that doesnt require me to perch my laptop on it so precariously. it just fell off. ']
```

To preprocess the raw data:

```bash
python3 src/preprocess.py
```

This will create `data/processed/` with cleaned text.

### 3️⃣ Model Training

Train the BERT model by running:

```bash
python3 src/train.py
```

This will generate:
- The trained model in `model/`
- Evaluation test data in `data/evaluation/`
- Training output (Logs and Checkpoints) in `results/`

### 4️⃣ Model Evaluation

Once training is complete, evaluate the model:

```bash
python3 src/evaluate.py
```

This will print a classification report and confusion matrix.

### 5️⃣ Inference (Single Message Prediction)

Run an interactive script to predict sentiment:

```bash
python3 src/inference.py
```

Example usage:
```
Enter a message: "I love this product!"
Prediction: Positive
```

To exit, type `exit`.

---

## 🔧 Configuration

Hyperparameters and paths are set in `src/constants.py`. Modify values as needed:

```python
BATCH_SIZE = 128
EPOCHS = 1
MAX_LEN = 128
LEARNING_RATE = [5e-4, 5e-5]
N_SPLITS = 2
```