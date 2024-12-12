import numpy as np
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

own_embs_npa = np.array(own_embeddings)

own_unk_embedding = np.mean(own_embs_npa, axis=0).tolist()

own_dim = own_embs_npa.shape[1]


own_sos_embedding = [0.0] * own_dim
own_sos_embedding[-3] = 1.0

own_eos_embedding = [0.0] * own_dim
own_eos_embedding[-2] = 1.0

own_pad_embedding = [0.0] * own_dim
own_pad_embedding[-4] = 1.0


# Update vocab and embeddings
own_vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + own_vocab
own_embeddings = [own_pad_embedding, own_sos_embedding, own_eos_embedding, own_unk_embedding] + own_embeddings

own_vocab_npa = np.array(own_vocab)
own_embs_npa = np.array(own_embeddings)

def tokenize(text):
    return text.lower().strip().split()
own_stoi_dict = {word: idx for idx, word in enumerate(own_vocab_npa)}
own_unk_idx = own_stoi_dict["<UNK>"]

def own_stoi(string):
    return own_stoi_dict.get(string, own_unk_idx)


def own_numericalize(text):
    tokenized_text = tokenize(text)
    return [
        own_stoi(token)
        for token in tokenized_text
    ]
if __name__ == '__main__':
    print(own_embs_npa.shape[1])