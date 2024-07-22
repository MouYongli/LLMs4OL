import re
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def advanced_similarity(str1, str2):
    # Tokenize the strings
    tokens1 = set(re.findall(r'\w+', str1))
    tokens2 = set(re.findall(r'\w+', str2))
    
    # Calculate Jaccard similarity for tokens
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    jaccard_similarity = len(intersection) / len(union)
    
    # Calculate overall similarity
    seq_similarity = similar(str1, str2)
    overall_similarity = (seq_similarity + jaccard_similarity) / 2
    return overall_similarity

# Strings to compare
string1 = "Intracellular"
string2 = "U4/U6 x U5 tri-snRNP complex"

# Calculate similarity
advanced_similarity_value = advanced_similarity(string1, string2)
print(advanced_similarity_value)
