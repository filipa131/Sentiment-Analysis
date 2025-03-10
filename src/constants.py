import os


# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Specific directories
DATA_DIR = os.path.join(BASE_DIR, 'data')

EVALUATION_DIR = os.path.join(DATA_DIR, 'evaluation')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
STOPWORDS_DIR = os.path.join(DATA_DIR, 'stopwords')

MODEL_DIR = os.path.join(BASE_DIR, 'model')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
SRC_DIR = os.path.join(BASE_DIR, 'src')


# File paths
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'processed_data.csv') 
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, 'data.csv.gz')  
STOPWORDS_FILE = os.path.join(STOPWORDS_DIR, 'stopwords_english.txt')  
TEST_FILE = os.path.join(EVALUATION_DIR, 'test.txt') 


# Source code paths
CLEANER_SCRIPT = os.path.join(SRC_DIR, 'cleaner.py')  
EVALUATE_SCRIPT = os.path.join(SRC_DIR, 'evaluate.py')  
INFERENCE_SCRIPT = os.path.join(SRC_DIR, 'inference.py')  
PREPROCESS_SCRIPT = os.path.join(SRC_DIR, 'preprocess.py') 
TRAIN_SCRIPT = os.path.join(SRC_DIR, 'train.py')  
UTILS_SCRIPT = os.path.join(SRC_DIR, 'utils.py') 


# Model paths
MODEL_PATH = os.path.join(MODEL_DIR, 'bert_model')  
TOKENIZER_PATH = os.path.join(MODEL_DIR, 'bert_tokenizer') 


# Model results
TEST_RESULTS_FILE = os.path.join(RESULTS_DIR, 'test_results.csv')  


# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 1 
LEARNING_RATE = [5e-4, 5e-5]
MAX_LEN = 128
N_SPLITS = 2


# Model version
MODEL_VERSION = "v1.0.0"  