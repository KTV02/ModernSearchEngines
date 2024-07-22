#check content of a given website (stored in txt) if it really contains the provided keywords
def check_keywords_in_document(document_path, keywords, encoding='utf-8'):
    """
    Checks if a document contains any of the given keywords and prints out which keywords are found.

    Args:
        document_path (str): The path to the document file.
        keywords (list): A list of keywords to check in the document.
        encoding (str): The encoding of the document file.
    """
    try:
        with open(document_path, 'r', encoding=encoding) as file:
            content = file.read()
        
        found_keywords = [keyword for keyword in keywords if keyword in content]
        
        if found_keywords:
            print("The document contains the following keywords:")
            for keyword in found_keywords:
                print(f"- {keyword}")
        else:
            print("No keywords found in the document.")
    
    except FileNotFoundError:
        print(f"Error: The file '{document_path}' was not found.")
    except UnicodeDecodeError as e:
        print(f"Unicode decode error: {e}. Try a different encoding.")
    except Exception as e:
        print(f"An error occurred: {e}")

#provide path to text file containing the website to check
document_path = 'website.txt'
#keywords to check
keywords = ['tübingen', 'tubingen', 'tuebingen', 'tuebing', 'tübing', 't%c3%bcbingen']
check_keywords_in_document(document_path, keywords)
