import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Read the text file
file_path = 'output.txt'

# Function to flatten nested lists
def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

# Function to parse the text file and return a list of tuples
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

# Parse the results
results = parse_results(file_path)

# Convert to DataFrame
df = pd.DataFrame(results, columns=['index', 'title', 'url', 'content', 'accuracy'])
print(df["title"])

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
best_by_topic_df = pd.concat(best_by_topic)

# Display the results
final_results = pd.concat([top_5_results, best_by_topic_df]).reset_index(drop=True)
final_results.to_csv('final_results.csv', index=False)
# Output the results
print(final_results)
