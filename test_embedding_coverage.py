from tqdm import tqdm

CAPTIONS_FILE = 'data/flickr8k/captions.txt'
EMBEDDING_FILE = 'embedding/glove-wiki-gigaword-100.txt'

import pandas as pd

with open(EMBEDDING_FILE, 'rt', encoding='utf-8') as fr:
    full_content = fr.read().strip().split('\n')

vocab = []
for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    vocab.append(i_word)


df = pd.read_csv(CAPTIONS_FILE)
df = list(df['caption'])

is_in, is_out = 0, 0
from collections import defaultdict
out_dict = defaultdict(int)

for caption in tqdm(df):
    for word in caption.lower().strip().split():
        if word in vocab:
            is_in += 1
        else:
            out_dict[word] += 1
            is_out += 1

print(is_in/(is_in + is_out))
print(sorted(out_dict.items(), key=lambda x: x[1]))