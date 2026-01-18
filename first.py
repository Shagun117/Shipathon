# from google.colab import files
# uploaded = files.upload()
import json
import re
import sys
sys.stdout.reconfigure(encoding="utf-8")

file_path = "chat_W.txt"
names = []
messages = []
current_name = None

pattern = r"""^
\d{1,2}/\d{1,2}/\d{2,4},\s      # date
\d{1,2}:\d{2}                  # time
(?:\:\d{2})?                   # optional seconds
\s?(?:AM|PM|am|pm)\s-\s        # AM/PM (any case)
([^:]+):\s                     # name
(.*)                            # message
"""

regex = re.compile(pattern, re.VERBOSE)
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        match = regex.match(line)

        if match:
            name = match.group(1)
            message = match.group(2)

            current_name = name
            names.append(name)
            messages.append(message)

        else:
            # multiline continuation
            if current_name and messages:
                messages[-1] += " " + line

# print("Names:", names)
# print("Messages:", messages)
 #Create empty dictionary
dict = {}

# Loop through both lists
for i in range(len(names)):
    name = names[i]
    msg = messages[i]

    # If name already exists, append message
    if name in dict:
        dict[name].append(msg)
    else:
        # Otherwise, create a new list with the message
        dict[name] = [msg]
with open("parsed_chat.json", "w", encoding="utf-8") as f:
    json.dump(dict, f, ensure_ascii=False, indent=2)



namesM = []
dialoguesM = []

with open("Script_M.txt", "r") as f:
    for line in f:
        line = line.strip()

        # skip empty lines or stage directions
        if not line or ":" not in line:
            continue

        name, dialogue = line.split(":", 1)

        namesM.append(name.strip())
        dialoguesM.append(dialogue.strip())

# print("Names = ", namesM)
# print("Dialogues = ", dialoguesM)

# Create empty dictionary
name_message_dict = {}

# Loop through both lists
for i in range(len(names)):
    name = namesM[i]
    msg = dialoguesM[i]

    # If name already exists, append message
    if name in name_message_dict:
        name_message_dict[name].append(msg)
    else:
        # Otherwise, create a new list with the message
        name_message_dict[name] = [msg]

# print(name_message_dict)


# pip install nltk scikit-learn sentence-transformers

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# ------------------------------------
# 1. TEXT PREPROCESSING FUNCTION
# ------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation/numbers
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# ------------------------------------
# 2. COMBINE DIALOGUES PER CHARACTER
# ------------------------------------
processed_texts = {}
for name, dialogues in dict.items():
    combined = " ".join(dialogues)
    processed_texts[name] = preprocess_text(combined)

# ------------------------------------
# 3. TF-IDF VECTORIZATION
# ------------------------------------
tfidf = TfidfVectorizer()
tfidf_vectors = tfidf.fit_transform(processed_texts.values())

# ------------------------------------
# 4. SENTENCE TRANSFORMER EMBEDDINGS
# ------------------------------------
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

character_embedding_w = {}
for name, text in processed_texts.items():
    embedding = model.encode(text)
    character_embedding_w[name] = embedding

# ------------------------------------
# OUTPUT
# ------------------------------------
# for name, vector in character_embedding_w.items():
    # print(f"{name} vector shape: {vector.shape}")

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# ------------------------------------
# 1. TEXT PREPROCESSING FUNCTION
# ------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation/numbers
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# ------------------------------------
# 2. COMBINE DIALOGUES PER CHARACTER
# ------------------------------------
processed_texts = {}
for name, dialogues in name_message_dict.items():
    combined = " ".join(dialogues)
    processed_texts[name] = preprocess_text(combined)

# ------------------------------------
# 3. TF-IDF VECTORIZATION
# ------------------------------------
tfidf = TfidfVectorizer()
tfidf_vectors = tfidf.fit_transform(processed_texts.values())

# ------------------------------------
# 4. SENTENCE TRANSFORMER EMBEDDINGS
# ------------------------------------
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

character_embedding_m = {}
for name, text in processed_texts.items():
    embedding = model.encode(text)
    character_embedding_m[name] = embedding

# ------------------------------------
# OUTPUT
# ------------------------------------
#for name, vector in character_embedding_m.items():
  #  print(f"{name} vector shape: {vector.shape}")


# Install (run once in Colab)
# !pip install -U sentence-transformers scikit-learn numpy

import re
import nltk
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# -------------------------------
# SIMPLE KEYWORD EXTRACTOR
# -------------------------------
STOP_WORDS = set(stopwords.words("english"))

def extract_keywords(text):
    if isinstance(text, list):
        text = " ".join(text)

    words = re.findall(r"[a-z]+", text.lower())

    return {
        stemmer.stem(w)
        for w in words
        if w not in STOP_WORDS and len(w) > 5
    }

# -------------------------------
# EXTRACT KEYWORDS
# -------------------------------
# ASSUMES THESE EXIST:
# dict
# name_message_dict
# MIN_SIMILARITY

keywords_A = {k: extract_keywords(v) for k, v in dict.items()}
keywords_B = {k: extract_keywords(v) for k, v in name_message_dict.items()}

# -------------------------------
# KEYWORD SIMILARITY (JACCARD)
# -------------------------------
def keyword_similarity(set1, set2):
    if not set1 or not set2:
        return 0
    return len(set1 & set2) / len(set1 | set2)

# -------------------------------
# MATCHING (ONE-TO-ONE)
# -------------------------------
results = {}
used_B = set()

for name_A, kw_A in keywords_A.items():
    best_match = None
    best_score = 0
    best_keywords = set()
    MIN_SIMILARITY=0.0001
    for name_B, kw_B in keywords_B.items():
        if name_B in used_B:
            continue

        score = keyword_similarity(kw_A, kw_B)

        if score > best_score:
            best_score = score
            best_match = name_B
            best_keywords = kw_A & kw_B

    if best_match and best_score >= MIN_SIMILARITY:
        used_B.add(best_match)
    else:
        best_match = None
        best_keywords = set()

    results[name_A] = {
        "best_match": best_match,
        "similarity": round(best_score, 3),
        "matched_keywords": list(best_keywords)
    }

# -------------------------------
# OUTPUT
# -------------------------------
for k, v in results.items():
    print(f"{k} -> {v['best_match']}")
    # print(f"  similarity: {v['similarity']}")
    # print(f"  matched keywords: {v['matched_keywords']}")

# print("Parsing started")
# print("Messages parsed:", len(messages))
