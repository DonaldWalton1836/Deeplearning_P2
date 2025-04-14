import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import requests
from collections import defaultdict

# -------------------------------
# STEP 1: Load BERT tokenizer and model
# -------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# -------------------------------
# STEP 2: Download and parse the dataset
# -------------------------------
url = "https://www.cs.fsu.edu/~liux/courses/deepRL/assignments/word-test.v1.txt"
response = requests.get(url)
lines = response.text.strip().split('\n')

group = "capital-common-countries"  # You can change to another group
analogies = []

in_group = False
for line in lines:
    if line.startswith(":"):
        in_group = group in line
        continue
    if in_group:
        parts = line.strip().split()
        if len(parts) == 4:
            analogies.append(parts)

# -------------------------------
# STEP 3: Get BERT embeddings for words
# -------------------------------
def get_word_embedding(word):
    tokens = tokenizer.tokenize(word)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([token_ids])
    with torch.no_grad():
        outputs = model(input_ids)
    embeddings = outputs.last_hidden_state[0]
    return embeddings.mean(dim=0)

# Collect all unique words in this group
word_set = set()
for a, b, c, d in analogies:
    word_set.update([a, b, c, d])

word_embeddings = {}
print("Generating embeddings...")
for word in word_set:
    word_embeddings[word] = get_word_embedding(word)
print(f"Loaded embeddings for {len(word_embeddings)} unique words.")

# -------------------------------
# STEP 4: Prediction and Evaluation
# -------------------------------
def cosine_similarity(v1, v2):
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

def l2_distance(v1, v2):
    return torch.norm(v1 - v2).item()

candidates = set()
for _, b, _, d in analogies:
    candidates.add(b)
    candidates.add(d)
candidates = list(candidates)

k_values = [1, 2, 5, 10, 20]
cosine_hits = defaultdict(int)
l2_hits = defaultdict(int)
total = 0

print("Evaluating analogies...")
for a, b, c, d_true in analogies:
    vec_a = word_embeddings[a]
    vec_b = word_embeddings[b]
    vec_c = word_embeddings[c]
    vec_d_true = word_embeddings[d_true]

    predicted_vec = vec_b - vec_a + vec_c

    cosine_scores = []
    l2_scores = []

    for candidate in candidates:
        if candidate in [a, b, c]:
            continue
        vec_d = word_embeddings[candidate]
        cos_sim = cosine_similarity(predicted_vec, vec_d)
        l2_dist = l2_distance(predicted_vec, vec_d)
        cosine_scores.append((candidate, cos_sim))
        l2_scores.append((candidate, l2_dist))

    cosine_scores.sort(key=lambda x: x[1], reverse=True)
    l2_scores.sort(key=lambda x: x[1])

    total += 1
    for k in k_values:
        top_k_cos = [word for word, _ in cosine_scores[:k]]
        top_k_l2 = [word for word, _ in l2_scores[:k]]
        if d_true in top_k_cos:
            cosine_hits[k] += 1
        if d_true in top_k_l2:
            l2_hits[k] += 1

# -------------------------------
# STEP 5: Print results
# -------------------------------
print(f"\nEvaluation Results for Group: {group}")
print(f"{'k':<4} {'Cosine Accuracy':<20} {'L2 Accuracy':<20}")
for k in k_values:
    cos_acc = cosine_hits[k] / total * 100
    l2_acc = l2_hits[k] / total * 100
    print(f"{k:<4} {cos_acc:<20.2f} {l2_acc:<20.2f}")
