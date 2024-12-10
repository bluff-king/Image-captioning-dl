import os

import torch
from torch import nn
import wandb
from torch.utils.data import DataLoader
from models.own_word_embedder import CenterWordPredictor
from dataset.own_embedder_data import get_mapped_data
from dataset.word_index_dict import description_tokens
from dataset.dataset import ContextToWordDataset
from dataset.numericalize_captions import NumericalizedDescriptions
import json
with open("config.json", "r") as json_file:
    cfg = json.load(json_file)


own_lstm_params= cfg['hyperparameters']['own_embedder_lstm']

EMBEDDING_DIMENSION = own_lstm_params['embedding_dim'] - 4 
LEARNING_RATE = own_lstm_params['learning_rate']
BATCH_SIZE = own_lstm_params['batch_size']
NUM_EPOCHS = own_lstm_params['num_epochs']

CHECKPOINT_PATH = cfg['paths']['checkpoint_path']
MODEL_EMBEDDING_PATH = f'{CHECKPOINT_PATH}word_embedder_model.pth'
MODEL_EMBEDDING_PATH_COPY = f'{CHECKPOINT_PATH}word_embedder_model_copy.pth'

all_description_indices = NumericalizedDescriptions()

all_mapping = get_mapped_data()
word_to_index_dict, index_to_word_dict = description_tokens(all_mapping)
vocabulary_size = len(word_to_index_dict)

all_dataset = ContextToWordDataset(all_description_indices, index_to_word_dict,
                word_to_index_dict, contextLength=3)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


wandb.init(
    project = "ProjectContextWords",
    config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "embedding_dim": EMBEDDING_DIMENSION,
        "optimizer" : "ADAM",
        "train_ratio":0.8,
        "val_ratio": 0.1,
})



word_embedder = CenterWordPredictor(vocabulary_size, EMBEDDING_DIMENSION)
if use_cuda:
    word_embedder = word_embedder.cuda()

word_embedder_parameters = filter(lambda p: p.requires_grad, word_embedder.parameters())

optimizer = torch.optim.Adam(word_embedder_parameters, lr=LEARNING_RATE)

# logit will be passed to lossFcn, inside will have softmax
lossFcn = nn.CrossEntropyLoss()
all_dataloader = DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=True) # when call, return one batch dataset
lowestTrainLoss = float('inf')


for epoch in range(1,NUM_EPOCHS):    
    word_embedder.train()
    loss_sum = 0.0
    number_of_batches = 0        
    for (context_around_idxs_Tsr, target_center_word_ndx) in all_dataloader:
        if number_of_batches % 50 == 0 and number_of_batches > 0:
            print (".", end="", flush=True)
        if use_cuda:
            context_around_idxs_Tsr = context_around_idxs_Tsr.cuda()
            target_center_word_ndx = target_center_word_ndx.cuda()
            
        predicted_center_word_ndx = word_embedder(context_around_idxs_Tsr)

        optimizer.zero_grad()
        loss = lossFcn(predicted_center_word_ndx, target_center_word_ndx)
        loss.backward()
        
        optimizer.step()
        
        loss_sum += loss.item()
        number_of_batches += 1
    train_loss = loss_sum/number_of_batches
    if train_loss < lowestTrainLoss:
        lowestTrainLoss = train_loss
        torch.save(word_embedder.state_dict(), MODEL_EMBEDDING_PATH)
    print ("\nepoch {}: train_loss = {}".format(epoch, train_loss))
    wandb.log({'Train_loss': train_loss})
wandb.finish()
