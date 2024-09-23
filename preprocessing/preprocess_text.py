import nltk
import re
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess a text string.
    """
    text = text.lower() # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split() # Split text into tokens
    tokens = [t for t in tokens if t not in stopwords.words('english')] # Remove stopwords
    return ' '.join(tokens) # Join tokens back into a string