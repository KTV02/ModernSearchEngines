# -*- coding: utf-8 -*-
# Performs topic modeling on the final output of the ranker 
# Output of this is used in UI as final rank and for Topic modeling graph

# Diversification is given by this formula:
# Combined Score=α×Relevance Score+(1−α)×Diversity Score

# 70% of score based on ranker score and 30 % based on diversity score
diversification_alpha = 0.7

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
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
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
                try:
                    content_list = ast.literal_eval(prefix_parts[3])
                except SyntaxError as e:
                    print(f"Syntax error when parsing content: {e}")
                    continue
                flat_content_list = flatten_list(content_list)  # Flatten nested lists
                content = ' '.join(flat_content_list)  # Combine content parts
                accuracy = float(accuracy_part)
                results.append((index, title, url, content, accuracy))
    return results

# Function to perform all calculations and store results
# This performs the topic modeling and stores the results
def perform_calculations(file_path, num_results=100, alpha=diversification_alpha):
    global final_results
    try:
        # Parse the results
        results = parse_results(file_path)
        if not results:
            print("No results were parsed.")
            return False

        # Convert to DataFrame
        global df
        df = pd.DataFrame(results, columns=['index', 'title', 'url', 'content', 'accuracy'])

        # Vectorize the content for LDA
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            X = vectorizer.fit_transform(df['content'])
        except ValueError as e:
            print(f"Value error during vectorization: {e}")
            return False

        # Perform LDA
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        try:
            lda.fit(X)
        except ValueError as e:
            print(f"Value error during LDA fitting: {e}")
            return False

        # Get topic distributions for each document
        try:
            topic_distributions = lda.transform(X)
        except ValueError as e:
            print(f"Value error during LDA transformation: {e}")
            return False

        # Add topic distributions to DataFrame
        for i in range(5):
            df[f'topic_{i}'] = topic_distributions[:, i]

        # Calculate diversity score (1 / (number of documents in topic + 1))
        df['diversity_score'] = df.apply(lambda row: sum([row[f'topic_{i}'] / (df[f'topic_{i}'].sum() + 1) for i in range(5)]), axis=1)
        
        # Calculate combined score
        df['combined_score'] = alpha * df['accuracy'] + (1 - alpha) * df['diversity_score']

        # Get the top results by combined score
        global top_results
        top_results = df.nlargest(num_results, 'combined_score')

        # Function to get the best result for a topic not in top results
        def get_best_for_topic(topic_num, selected_urls):
            topic_col = f'topic_{topic_num}'
            filtered_df = df[~df.index.isin(top_results.index)]
            filtered_df = filtered_df[~filtered_df['url'].isin(selected_urls)]
            if filtered_df.empty:
                return pd.DataFrame()
            best_result = filtered_df.nlargest(1, 'combined_score')
            return best_result

        # Collect the best result for each topic
        selected_urls = set(top_results['url'])
        best_by_topic = []
        for topic_num in range(5):
            best_result = get_best_for_topic(topic_num, selected_urls)
            if not best_result.empty:
                selected_urls.add(best_result.iloc[0]['url'])
                best_by_topic.append(best_result)

        # Concatenate results into a single DataFrame
        global best_by_topic_df
        if best_by_topic:
            best_by_topic_df = pd.concat(best_by_topic)
            final_results = pd.concat([top_results, best_by_topic_df]).reset_index(drop=True)
        else:
            final_results = top_results.reset_index(drop=True)
        
        # Save the final results to a CSV file
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

# Example usage: Output of ranker in this case would be topicmodelingoutput.txt
file_path = 'topicmodelingoutput.txt'
num_results = 100 # Change this to the number of search results you want
if perform_calculations(file_path, num_results):
    search_results = get_search_results()
    print("Search Results:")
    print(search_results)

    topic_arrays = get_topic_arrays()
    print("Topic Arrays:")
    for i, topic_array in enumerate(topic_arrays):
        print(f"Topic {i} Array:")
        for doc in topic_array:
            print(doc)
