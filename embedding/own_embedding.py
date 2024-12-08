import numpy as np
import torch
from models.own_word_embedder import CenterWordPredictor
from dataset.word_index_dict import description_tokens

import json
with open("config.json", "r") as json_file:
    cfg = json.load(json_file)
CHECKPOINT_PATH = cfg['paths']['checkpoint_path']

EMBEDDING_DIMENSION = 124  

MODEL_EMBEDDING_PATH = f'{CHECKPOINT_PATH}word_embedder_model.pth'
MODEL_EMBEDDING_PATH_COPY = f'{CHECKPOINT_PATH}word_embedder_model_copy.pth'

word_to_index_dict, index_to_word_dict = description_tokens()
vocabulary_size = len(word_to_index_dict)


word_embedder = CenterWordPredictor(vocabulary_size, EMBEDDING_DIMENSION)
word_embedder.load_state_dict(torch.load(MODEL_EMBEDDING_PATH_COPY,weights_only=True))

word_embeddings = word_embedder.embedding.weight.data.cpu().numpy() 
word_to_embedding = {word: word_embeddings[index] for word, index in word_to_index_dict.items()}

current_vocab_size, current_embedding_dim = word_embeddings.shape

vocab, ebd = zip(*word_to_embedding.items())
embeddings = []
vocab = list(vocab)  
ebd = list(ebd)
embeddings = []
for i in range(len(ebd)):
    i_embeddings = list(ebd[i])  
    i_embeddings.extend([0.0, 0.0, 0.0, 0.0]) 
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
embeddings = [pad_embedding, sos_embedding, eos_embedding, unk_embedding] + embeddings

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