import emoji
import re

from constants import STOPWORDS_FILE

class Cleaner:
    """
    A class for cleaning and processing text messages for BERT sentiment analysis.
    """

    def __init__(self) -> None:
        """
        Initializes the Cleaner by loading stop words from a file.
        """
        with open(STOPWORDS_FILE, 'r') as file:
            stop_words_list = file.read().splitlines()
        self.stop_set = set(stop_words_list)

    def remove_urls_mentions(self, content: str) -> str:
        """
        Removes URLs and @mentions from the text.
        """
        content = re.sub(r"http\S+|www\S+|https\S+", " [URL] ", content, flags=re.MULTILINE)  # Replace URLs
        content = re.sub(r"@\w+", " [MENTION] ", content)  # Replace @mentions
        return content

    def normalize_punctuation(self, content: str) -> str:
        """
        Normalizes punctuation, keeping only key sentiment-carrying punctuation (? . ! ,).
        Removes all other punctuation.
        """
        content = re.sub(r'([?.!,])', r' \1 ', content)  # Keep spaces around important punctuation 
        content = re.sub(r'[^a-zA-Z0-9?.!, ]', '', content)  # Remove all other punctuation (except ? . ! ,)
        content = re.sub(r'\s+', ' ', content).strip()  # Remove extra spaces

        return content

    def replace_emojis(self, content: str) -> str:
        """
        Converts emojis to text descriptions.
        """
        return emoji.demojize(content, delimiters=(" ", " "))  # 🙂 → " slightly_smiling_face "

    def remove_stopwords(self, content: str) -> str:
        """
        Removes stopwords from the content.
        """
        words = content.split()  
        filtered_words = [word for word in words if word not in self.stop_set]  
        return ' '.join(filtered_words)

    def preprocess_message(self, content: str) -> str:
        """
        Preprocesses a message for BERT-based sentiment analysis.
        """
        content = str(content).lower()  # Lowercase for using 'bert-base-uncased'
        content = self.remove_urls_mentions(content)  # Remove URLs and @mentions
        content = self.replace_emojis(content)  # Convert emojis to text
        content = self.normalize_punctuation(content)  # Normalize punctuation spacing
        content = self.remove_stopwords(content)  # Remove stopwords

        return content
