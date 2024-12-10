import numpy as np
import shiti;
import torch
from models.own_word_embedder import CenterWordPredictor
from dataset.word_index_dict import description_tokens

import json
with open("config.json", "r") as json_file:
    cfg = json.load(json_file)

own_lstm_params= cfg['hyperparameters']['own_embedder_lstm']
CHECKPOINT_PATH = cfg['paths']['checkpoint_path']
EMBEDDING_FILE = cfg['paths']['embedding_file']
EMBEDDING_DIMENSION = own_lstm_params['embedding_dim'] - 4 

MODEL_EMBEDDING_PATH = f'{CHECKPOINT_PATH}word_embedder_model.pth'
# Copy file
shutil.copy(MODEL_EMBEDDING_PATH, MODEL_EMBEDDING_PATH_COPY)
print(f"File was copied {MODEL_EMBEDDING_PATH} to {MODEL_EMBEDDING_PATH_COPY}")

MODEL_EMBEDDING_PATH_COPY = f'{CHECKPOINT_PATH}word_embedder_model_copy.pth'

word_to_index_dict, index_to_word_dict = description_tokens()
vocabulary_size = len(word_to_index_dict)

word_embedder = CenterWordPredictor(vocabulary_size, EMBEDDING_DIMENSION)
word_embedder.load_state_dict(torch.load(MODEL_EMBEDDING_PATH_COPY,weights_only=True))

word_embeddings = word_embedder.embedding.weight.data.cpu().numpy() 
word_to_embedding = {word: word_embeddings[index] for word, index in word_to_index_dict.items()}

own_vocab, own_ebd = zip(*word_to_embedding.items())
own_vocab = list(own_vocab)  
own_ebd = list(own_ebd)

own_embeddings = []
for i in range(len(own_ebd)):
    i_own_embeddings = list(own_ebd[i])  
    i_own_embeddings.extend([0.0, 0.0, 0.0, 0.0]) 
    own_embeddings.append(i_own_embeddings)


vocab, embeddings = [], []
with open(EMBEDDING_FILE, 'rt', encoding='utf-8') as fr:
    full_content = fr.read().strip().split('\n')

for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    i_embeddings.extend([0.0, 0.0, 0.0, 0.0])
    vocab.append(i_word)
    embeddings.append(i_embeddings)

embs_npa = np.array(embeddings)
own_embs_npa = np.array(own_embeddings)

unk_embedding = np.mean(embs_npa, axis=0).tolist()
own_unk_embedding = np.mean(own_embs_npa, axis=0).tolist()

dim = embs_npa.shape[1]
own_dim = own_embs_npa.shape[1]

sos_embedding = [0.0] * dim
sos_embedding[-3] = 1.0
own_sos_embedding = [0.0] * own_dim
own_sos_embedding[-3] = 1.0

eos_embedding = [0.0] * dim
eos_embedding[-2] = 1.0
own_eos_embedding = [0.0] * own_dim
own_eos_embedding[-2] = 1.0

pad_embedding = [0.0] * dim
pad_embedding[-4] = 1.0
own_pad_embedding = [0.0] * own_dim
own_pad_embedding[-4] = 1.0


# Update vocab and embeddings
vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + vocab
own_vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + own_vocab
embeddings = [pad_embedding, sos_embedding, eos_embedding, unk_embedding] + embeddings
own_embeddings = [own_pad_embedding, own_sos_embedding, own_eos_embedding, own_unk_embedding] + own_embeddings

vocab_npa = np.array(vocab)
own_vocab_npa = np.array(own_vocab)
embs_npa = np.array(embeddings)
own_embs_npa = np.array(own_embeddings)

def tokenize(text):
    return text.lower().strip().split()

stoi_dict = {word: idx for idx, word in enumerate(vocab_npa)}
_unk_idx = stoi_dict["<UNK>"]
own_stoi_dict = {word: idx for idx, word in enumerate(own_vocab_npa)}
own_unk_idx = own_stoi_dict["<UNK>"]

def own_stoi(string):
    return own_stoi_dict.get(string, own_unk_idx)

def stoi(string):
    return stoi_dict.get(string, _unk_idx)


def own_numericalize(text):
    tokenized_text = tokenize(text)
    return [
        own_stoi(token)
        for token in tokenized_text
    ]
def numericalize(text):
    tokenized_text = tokenize(text)
    return [
        stoi(token)
        for token in tokenized_text
    ]


if __name__ == '__main__':
    print(own_embs_npa.shape[1])
