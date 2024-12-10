import numpy as np
import json

with open("config.json", "r") as json_file:
    cfg = json.load(json_file)


EMBEDDING_FILE = cfg['paths']['embedding_file']


vocab, embeddings = [], []
with open(EMBEDDING_FILE, 'rt', encoding='utf-8') as fr:
    full_content = fr.read().strip().split('\n')

for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    i_embeddings.extend([0.0, 0.0, 0.0, 0.0])
    vocab.append(i_word)
    embeddings.append(i_embeddings)

vocab_npa = np.array(vocab)
embs_npa = np.array(embeddings)

unk_embedding = np.mean(embs_npa, axis=0).tolist()

dim = embs_npa.shape[1]
sos_embedding = [0.0] * dim
sos_embedding[-3] = 1.0
eos_embedding = [0.0] * dim
eos_embedding[-2] = 1.0
pad_embedding = [0.0] * dim
pad_embedding[-4] = 1.0
# unk_embedding = [0.0] * dim
# unk_embedding[-1] = 1.0

# Update vocab and embeddings
vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + vocab
embeddings = [pad_embedding, sos_embedding,
              eos_embedding, unk_embedding] + embeddings

vocab_npa = np.array(vocab)
embs_npa = np.array(embeddings)


def tokenize(text):
    return text.lower().strip().split()


stoi_dict = {word: idx for idx, word in enumerate(vocab_npa)}
_unk_idx = stoi_dict["<UNK>"]


def stoi(string):
    return stoi_dict.get(string, _unk_idx)


def numericalize(text):
    tokenized_text = tokenize(text)
    return [
        stoi(token)
        for token in tokenized_text
    ]


if __name__ == '__main__':
    print(embs_npa.shape[0])