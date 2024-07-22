#Performs topic modelling on the final output of the ranker 
#Ouput of this is used in UI as final rank and for Topic modeling graph

import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Function to flatten nested lists of the output format of the ranker (content of webpages)
def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

# Function to parse output of the ranker and return a list of tuples(index,title,url,content,score)
def parse_results(file_path):
    results = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                # Find the position of the last comma to split content and accuracy
                last_comma_index = line.rfind(',')
                if last_comma_index == -1:
                    continue

                # Extract the part before the last comma
                prefix = line[:last_comma_index]

                # Extract the accuracy part
                accuracy_part = line[last_comma_index + 1:].strip().strip('[]')
                
                # Split the prefix into components
                prefix_parts = prefix.split(', ', 3)  # Split the prefix into 4 parts: Index, Title, Url, Content
                
                index = int(prefix_parts[0].strip('['))
                title = prefix_parts[1].strip("'")
                url = prefix_parts[2].strip("'")
                content_list = ast.literal_eval(prefix_parts[3])
                flat_content_list = flatten_list(content_list)  # Flatten nested lists
                content = ' '.join(flat_content_list)  # Combine content parts
                accuracy = float(accuracy_part)
                results.append((index, title, url, content, accuracy))
    return results

# Function to perform all calculations and store results
#This performs the topic modeling and stores the results
def perform_calculations(file_path):
    try:
        # Parse the results
        results = parse_results(file_path)

        # Convert to DataFrame
        global df
        df = pd.DataFrame(results, columns=['index', 'title', 'url', 'content', 'accuracy'])

        # Vectorize the content for LDA
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df['content'])

        # Perform LDA
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(X)

        # Get topic distributions for each document
        topic_distributions = lda.transform(X)

        # Add topic distributions to DataFrame
        for i in range(5):
            df[f'topic_{i}'] = topic_distributions[:, i]

        # Get the top 5 results by accuracy
        global top_5_results
        top_5_results = df.nlargest(5, 'accuracy')

        # Function to get the best result for a topic not in top 5
        def get_best_for_topic(topic_num, selected_urls):
            topic_col = f'topic_{topic_num}'
            filtered_df = df[~df.index.isin(top_5_results.index)]
            filtered_df = filtered_df[~filtered_df['url'].isin(selected_urls)]
            best_result = filtered_df.nlargest(1, 'accuracy')
            return best_result

        # Collect the best result for each topic
        selected_urls = set(top_5_results['url'])
        best_by_topic = []
        for topic_num in range(5):
            best_result = get_best_for_topic(topic_num, selected_urls)
            if not best_result.empty:
                selected_urls.add(best_result.iloc[0]['url'])
                best_by_topic.append(best_result)

        # Concatenate results into a single DataFrame
        global best_by_topic_df
        best_by_topic_df = pd.concat(best_by_topic)

        # Display the results
        global final_results
        final_results = pd.concat([top_5_results, best_by_topic_df]).reset_index(drop=True)
        final_results.to_csv('final_results.csv', index=False)
        
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Function to get the search results to display
def get_search_results():
    return final_results

# Function to get the 5 arrays for the 5 topics created
def get_topic_arrays():
    selected_urls = set()
    topic_docs = []
    for topic_num in range(5):
        topic_col = f'topic_{topic_num}'
        topic_top_docs = df[~df['url'].isin(selected_urls)].nlargest(5, topic_col)[['title', 'url', 'accuracy']]
        topic_docs.append(topic_top_docs)
        selected_urls.update(topic_top_docs['url'])

    # Create arrays for each topic containing title, url, and accuracy
    topic_arrays = []
    for docs in topic_docs:
        topic_array = docs[['title', 'url', 'accuracy']].values.tolist()
        topic_arrays.append(topic_array)

    return topic_arrays


#Example usage: Output of ranker in this case would be output.txt
file_path = 'output.txt'
if perform_calculations(file_path):
    search_results = get_search_results()
    print("Search Results:")
    print(search_results)

    topic_arrays = get_topic_arrays()
    print("Topic Arrays:")
    for i, topic_array in enumerate(topic_arrays):
        print(f"Topic {i} Array:")
        for doc in topic_array:
            print(doc)
