import os
import pandas as pd

from cleaner import Cleaner
from constants import RAW_DATA_FILE, PROCESSED_DATA_DIR, PROCESSED_DATA_FILE
from utils import read_and_parse_file

# Load cleaner
cleaner = Cleaner()

# Create a DataFrame
columns = ["timestamp", "msgid", "sender", "flag", "target", "content"]
df = pd.DataFrame(read_and_parse_file(RAW_DATA_FILE), columns=columns)

# Preprocess data
df = df.astype(str)
df = df.drop_duplicates(subset=['content'])              
df['content_processed'] = df['content'].apply(cleaner.preprocess_message)


# Export preprocessed data
try:
    os.makedirs(PROCESSED_DATA_DIR)
except FileExistsError:
    pass

df[['content', 'content_processed', 'target']].to_csv(PROCESSED_DATA_FILE, index = False, sep = '|')