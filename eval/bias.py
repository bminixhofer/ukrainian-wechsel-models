from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import numpy as np

model_name_or_path = "checkpoint-250000"

bidict = {}

for line in open("../../wechsel/dicts/data/ukrainian.txt"):
    english, ukrainian = line.strip().split("\t")
    bidict[english] = ukrainian

lex_dict = {}

for line in open("vader_lexicon.txt"):
    (word, measure) = line.strip().split("\t")[0:2]
    if word in bidict:
        lex_dict[bidict[word]] = float(measure)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
embeddings = (
    AutoModel.from_pretrained(model_name_or_path)
    .get_input_embeddings()
    .weight.detach()
    .numpy()
)

negative_embs = []
positive_embs = []

for word, measure in lex_dict.items():
    tokens = tokenizer.encode(word, add_special_tokens=False)
    # if len(tokens) > 1:
    #     continue

    token = tokens[0]
    emb = embeddings[token]

    if measure > 0:
        positive_embs.append(emb)
    elif measure < 0:
        negative_embs.append(emb)

positive_embs = [embeddings[tokenizer.encode("погано")[1]]]
negative_embs = [embeddings[tokenizer.encode("добре")[1]]]

query = "російський"
query = "український"
# query = "європейський"
# query = "американський"

tokens = tokenizer.encode(query, add_special_tokens=False)
# if len(tokens) > 1:
#     raise NotImplementedError()

token = tokens[0]

positive_sim = cosine_similarity([embeddings[token]], positive_embs).mean()
negative_sim = cosine_similarity([embeddings[token]], negative_embs).mean()

print(positive_sim - negative_sim)
