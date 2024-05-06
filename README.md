# nlp_lab1

                               NLP LAB 1

Ankit Rathwa - 202001190

code :

from google.colab import drive
drive.mount('/content/drive')



import tarfile
import os


# Path to the .tgz file in Google Drive
path_to_tgz = '/content/drive/MyDrive/nlp/cnn_stories.tgz'


# Path to save the extracted files
extracted_folder_path = '/content/Extracted/'


# Create the extracted folder if it doesn't exist
os.makedirs(extracted_folder_path, exist_ok=True)


# Extract the .tgz file
with tarfile.open(path_to_tgz, 'r:gz') as tar:
    tar.extractall(extracted_folder_path)

mport tarfile
import os


# Path to the .tgz file in Google Drive
path_to_tgz = '/content/drive/MyDrive/nlp/dailymail_stories.tgz'


# Path to save the extracted files
extracted_folder_path = '/content/Extracted/'


# Create the extracted folder if it doesn't exist
os.makedirs(extracted_folder_path, exist_ok=True)


# Extract the .tgz file
with tarfile.open(path_to_tgz, 'r:gz') as tar:
    tar.extractall(extracted_folder_path)

mport os


# Path to the extracted folder
extracted_folder_path = '/content/Extracted/cnn/stories/'


# List files in the extracted folder
extracted_files = os.listdir(extracted_folder_path)


# Print the list of extracted files
print("Extracted Files:")
for file_name in extracted_files:
    print(file_name)

import os


# Path to the extracted folder
extracted_folder_path = '/content/Extracted/dailymail/stories/'


# List files in the extracted folder
extracted_files = os.listdir(extracted_folder_path)


# Print the list of extracted files
print("Extracted Files:")
for file_name in extracted_files:
    print(file_name)


import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool


# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")


# Function to extract noun phrases using spaCy
def extract_noun_phrases(text):
    doc = nlp(text)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    return ' '.join(noun_phrases)


# Function to process a single document and extract noun phrases
def process_document(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        noun_phrases_text = extract_noun_phrases(text)
        return noun_phrases_text


# Function to compute cosine similarity between noun phrases
def compute_cosine_similarity(noun_phrases_list):
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()


    # Fit and transform the text to obtain the TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(noun_phrases_list)


    # Compute cosine similarity between all pairs of noun phrases
    cosine_similarities = cosine_similarity(tfidf_matrix)
    return cosine_similarities


# Path to the directory containing the documents
documents_dir1 = '/content/Extracted/cnn/stories'


# List files in the directory and select the first 1000 files
file_names1 = [os.path.join(documents_dir1, file_name) for file_name in os.listdir(documents_dir1) if file_name.endswith('.story')][:1000]


# Process documents and extract noun phrases using parallel processing
with Pool() as pool:
    noun_phrases_list1 = pool.map(process_document, file_names1)


# Output all extracted noun phrases
print("Extracted Noun Phrases:")
for noun_phrases_text in noun_phrases_list1:
    print(noun_phrases_text)

# Compute cosine similarities between all pairs of noun phrases
cosine_similarities1 = compute_cosine_similarity(noun_phrases_list1)


# Output cosine similarities
print("\nCosine Similarities:")
for i in range(len(cosine_similarities1)):
    for j in range(i+1, len(cosine_similarities1[i])):
        similarity = cosine_similarities1[i, j]
        print(f"Cosine similarity between noun phrases {i} and {j}: {similarity}")


import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool


# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")


# Function to extract noun phrases using spaCy
def extract_noun_phrases(text):
    doc = nlp(text)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    return ' '.join(noun_phrases)


# Function to process a single document and extract noun phrases
def process_document(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        noun_phrases_text = extract_noun_phrases(text)
        return noun_phrases_text


# Function to compute cosine similarity between noun phrases
def compute_cosine_similarity(noun_phrases_list):
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()


    # Fit and transform the text to obtain the TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(noun_phrases_list)


    # Compute cosine similarity between all pairs of noun phrases
    cosine_similarities = cosine_similarity(tfidf_matrix)
    return cosine_similarities


# Path to the directory containing the documents
documents_dir2 = '/content/Extracted/dailymail/stories'


# List files in the directory and select the first 1000 files
file_names2 = [os.path.join(documents_dir2, file_name) for file_name in os.listdir(documents_dir2) if file_name.endswith('.story')][:1000]


# Process documents and extract noun phrases using parallel processing
with Pool() as pool:
    noun_phrases_list2 = pool.map(process_document, file_names2)


# Output all extracted noun phrases
print("Extracted Noun Phrases:")
for noun_phrases_text in noun_phrases_list2:
    print(noun_phrases_text)

# Compute cosine similarities between all pairs of noun phrases
cosine_similarities2 = compute_cosine_similarity(noun_phrases_list2)


# Output cosine similarities
print("\nCosine Similarities:")
for i in range(len(cosine_similarities2)):
    for j in range(i+1, len(cosine_similarities2[i])):
        similarity = cosine_similarities2[i, j]
        print(f"Cosine similarity between noun phrases {i} and {j}: {similarity}")






Outputs for noun-phraases in cnn stories : 



Outputs for cosine similarities among noun-phraases in cnn stories : 








Outputs for noun-phraases in dailymail stories : 



Outputs for cosine similarities among noun-phraases in dailmail stories : 



