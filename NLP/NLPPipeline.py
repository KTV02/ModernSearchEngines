import numpy as np
import nltk
import string
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def detect_language(text):
    """Detect the language of the given text."""
    try:
        return detect(text)
    except Exception as e:
        return str(e)

def remove_punctuation_and_tokenize(text):
    """Remove punctuation from the text and tokenize it."""
    translator = str.maketrans('', '', string.punctuation)
    text_no_punct = text.translate(translator)
    tokens = word_tokenize(text_no_punct)
    return tokens

def remove_stop_words(tokens, language='english'):
    """Remove stop words from the list of tokens."""
    stop_words = set(stopwords.words(language))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def pos_tagging(tokens):
    """Perform POS tagging on the list of tokens."""
    tagged_tokens = pos_tag(tokens)
    return tagged_tokens

#Lemmatizer using wordnet
def lemmatize_tokens(tokens):
    """Lemmatize the list of tokens."""
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# Example usage:
if __name__ == "__main__":
    text = "Tübingen (German: [ˈtyːbɪŋən] ⓘ; Swabian: Dibenga) is a traditional university city in central Baden-Württemberg, Germany. It is situated 30 km (19 mi) south of the state capital, Stuttgart, and developed on both sides of the Neckar and Ammer rivers. As of 2014[3] about one in three of the 90,000 people[citation needed] living in Tübingen is a student. As of the 2018/2019 winter semester, 27,665 students attend the Eberhard Karl University of Tübingen.[citation needed] The city has the lowest median age in Germany, in part due to its status as a university city. As of December 31, 2015, the average age of a citizen of Tübingen is 39.1 years."

    # Language Detection
    language = detect_language(text)
    print("Detected Language:", language)

    # Punctuation Removal and Tokenization
    tokens = remove_punctuation_and_tokenize(text)
    print("Tokens:", tokens)

    # Stop Word Removal
    filtered_tokens = remove_stop_words(tokens)
    print("Filtered Tokens:", filtered_tokens)

    # POS Tagging
    pos_tags = pos_tagging(filtered_tokens)
    print("POS Tags:", pos_tags)

    # Lemmatization
    lemmatized_tokens = lemmatize_tokens(filtered_tokens)
    print("Lemmatized Tokens:", lemmatized_tokens)
