"""
Vector-Space-Model + Intersection Algorithm
"""
from collections import defaultdict
import math
from vectorspace import VectorSpaceModel

"""
Extend Vector Space Model with Dates feature + Rating as Parametric indices.

Can be extended with other features. Rating is an example for numeric features. Dates for a date feature. 
"""
class ExtendedVectorSpaceModel(VectorSpaceModel):
    def __init__(self):
        super().__init__()
        self.feature_indices = defaultdict(list)
        self.numeric_indices = defaultdict(list)
    
    # Override the add_document method to include additional features like dates and numeric values
    def add_document(self, doc_id, text, **features):
        super().add_document(doc_id, text)
        for feature, value in features.items():
            if isinstance(value, (int, float)):
                self.numeric_indices[(feature, value)].append(doc_id)
            else:
                self.feature_indices[(feature, value)].append(doc_id)

    # New method to intersect postings for features
    def intersect_postings(self, *posting_lists):
        if not posting_lists:
            return []
        
        # Start with the first list and intersect with each subsequent list
        intersected = set(posting_lists[0])
        for postings in posting_lists[1:]:
            intersected.intersection_update(postings)
        
        return list(intersected)

    # New method to handle numeric range queries
    def range_query(self, feature, min_val, max_val):
        result = set()
        for value in range(math.floor(min_val), math.ceil(max_val) + 1):
            result.update(self.numeric_indices.get((feature, value), []))
        return result

    # Override the query method to include feature-based intersection and numeric range queries
    def query(self, query_text, numeric_ranges=None, **features):
        # Perform the standard VSM query
        base_results = super().query(query_text)
        
        # Collect document IDs for the additional feature filters
        feature_postings = []
        for feature, value in features.items():
            feature_postings.append(set(self.feature_indices.get((feature, value), [])))
        
        # Intersect the base results with feature postings
        if feature_postings:
            filtered_results = self.intersect_postings(set(base_results.keys()), *feature_postings)
        else:
            filtered_results = set(base_results.keys())
        
        # Handle numeric range queries
        if numeric_ranges:
            for feature, (min_val, max_val) in numeric_ranges.items():
                numeric_postings = self.range_query(feature, min_val, max_val)
                filtered_results = set(filtered_results).intersection(numeric_postings)
        
        final_results = {doc_id: base_results[doc_id] for doc_id in filtered_results if doc_id in base_results}
        
        return final_results

# Example usage
vsm = ExtendedVectorSpaceModel()

# Add documents with additional features like date and rating
vsm.add_document(0, "When Antony saw that Julius Caesar lay dead", date="2023-01-01", rating=5)
vsm.add_document(1, "The world saw the demise of Julius Caesar", date="2023-01-02", rating=4)
vsm.add_document(2, "Antony saw Julius Caesar lay dead", date="2023-01-01", rating=4)
vsm.add_document(3, "Napoleon Bonaparte was a French military leader", date="2023-02-01", rating=3)
vsm.add_document(4, "Alexander the Great conquered much of the known world", date="2023-02-01", rating=4)

# Build the model (vocabulary, TF-IDF vectors, etc.)
vsm._build_vocab()

# Query the model with additional feature filtering and numeric range queries
query_text = "Julius Caesar"
results = vsm.query(query_text, numeric_ranges={"rating": (4, 5)}, date="2023-01-01")

# Print results
print("Query Results for terms '{}', date '{}', and rating range (4, 5):".format(query_text, "2023-01-01"))
for doc_id, score in results.items():
    print(f"Document {doc_id} has score {score:.4f}")