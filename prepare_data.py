import glob
from bs4 import BeautifulSoup
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
import multiprocessing

total_bytes = 0
max_bytes = 100_000_000
articles = []

for path in tqdm(sorted(glob.glob("text/**/*"))):
    soup = BeautifulSoup(open(path).read(), features="lxml")

    for doc in soup.find_all("doc"):
        total_bytes += len(doc.text.encode("utf-8"))

        if total_bytes >= max_bytes:
            break

        articles.append(doc.text)

    if total_bytes >= max_bytes:
        break

lsh = MinHashLSH(threshold=0.5, num_perm=128)

for i, article in tqdm(enumerate(articles), total=len(articles)):
    h = MinHash()
    for word in article.split():
        h.update(word.encode('utf8'))

    lsh.insert(str(i), h)

def n_overlap(example):
    h = MinHash()
    for word in example["text"].split():
        h.update(word.encode('utf8'))

    return len(lsh.query(h))

train_dataset = load_dataset("oscar", "unshuffled_deduplicated_uk", split="train")
train_dataset = train_dataset.filter(lambda example: n_overlap(example) == 0, num_proc=multiprocessing.cpu_count())
train_dataset.save_to_disk("train_data")
open("valid_data.txt", "w").write("\n".join(articles))