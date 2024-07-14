import sys
import os
from collections import Counter, OrderedDict

# Add the directories to the system path
sys.path.append(os.path.abspath('../retriever/'))

# Import MyClass from lib/nested/my_class
from retriever.vectorspace import VectorSpaceModel

#can read the NLPOutput.txt file and returns the contents (as token lists), urls and titles used for ranking
def parse_tokens(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError encountered: {e}")
        return []

    tokens_list = []
    current_tokens = []
    titles = []
    urls =[]

    for line in lines:
        try:
            if line.startswith("Tokens: "):
                if current_tokens:
                    tokens_list.append(current_tokens)
                    current_tokens = []
                current_tokens.extend(line[len("Tokens: "):].strip().split())
            else:
                # Continue collecting tokens if we are in the middle of a tokens block
                if not line.startswith(" "):
                    if line.startswith("Title: "):
                        titles.append(line[len("Title: "):].strip())
                    elif line.startswith("URL: "):
                        urls.append(line[len("URL: "):].strip())
                    else:
                        current_tokens.extend(line.strip().split())

        except Exception as e:
            print(f"Error processing line: {line}. Error: {e}")

    # Append the last collected tokens, if any
    if current_tokens:
        tokens_list.append(current_tokens)

    return tokens_list, titles, urls

def get_ordered_word_frequency(corpus):
    """
    Process the corpus to count word frequencies and order them.

    :param corpus: List of documents, where each document is a string.
    :return: Ordered dictionary of words sorted by frequency in descending order.
    """
    # Combine all documents into a single list of words
    combined_bow = []
    for document in corpus:
        combined_bow.extend(document)

    # Count occurrences of each word in the combined list
    word_counts = Counter(combined_bow)

    # Sort the words by frequency in descending order and create an ordered dictionary
    ordered_word_counts = OrderedDict(word_counts.most_common())

    return ordered_word_counts


if __name__ == "__main__":
    tokens_list, _, _2 = parse_tokens("NLPOutput.txt")
    print(len(tokens_list), len(_), len(_2))
    #merged_tokens_list = [' '.join(tokens) for tokens in tokens_list]
    ordered_word_freq = get_ordered_word_frequency(tokens_list)
    with open('Keywords.txt', 'w', encoding='utf-8') as file:
        for word, freq in ordered_word_freq.items():
            file.write(f"{word}: {freq}\n")