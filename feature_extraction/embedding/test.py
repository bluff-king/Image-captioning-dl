import numpy as np


vocab, embeddings = [], []
with open('embedding/glove-wiki-gigaword-100.txt', 'rt') as fr:
    full_content = fr.read().strip().split('\n')

for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    i_embeddings.extend([0.0, 0.0, 0.0])
    vocab.append(i_word)
    embeddings.append(i_embeddings)

vocab_npa = np.array(vocab)
embs_npa = np.array(embeddings)

unk_embedding = np.mean(embs_npa, axis=0).tolist()

dim = embs_npa.shape[1]
sos_embedding = [0.0] * dim
sos_embedding[-2] = 1.0
eos_embedding = [0.0] * dim
eos_embedding[-1] = 1.0
pad_embedding = [0.0] * dim
pad_embedding[-3] = 1.0

# Update vocab and embeddings
vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + vocab
embeddings = [pad_embedding, sos_embedding, eos_embedding, unk_embedding] + embeddings

print(pad_embedding, sos_embedding, eos_embedding, unk_embedding, sep='\n')

vocab_npa = np.array(vocab)
embs_npa = np.array(embeddings)


print("Updated vocab size:", len(vocab_npa))
print("Updated embedding shape:", embs_npa.shape)