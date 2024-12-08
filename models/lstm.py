import torch
from torch import nn
import numpy as np
from torchvision import models
from embedding.embedding import embs_npa, vocab_npa, stoi

import json

with open("config.json", "r") as json_file:
    cfg = json.load(json_file)
    
lstm_params = cfg['hyperparameters']['glove_lstm']
CAPTIONS_LENGTH = lstm_params['captions_length']
HIDDEN_SIZE = lstm_params['hidden_dim']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pad_idx = stoi("<PAD>")


class ImageCaptioningLstm(nn.Module):
    def __init__(self, embed_size = embs_npa.shape[1], vocab_size = len(vocab_npa), hidden_size = HIDDEN_SIZE,pad_idx = pad_idx, dropout = 0.5):
        super(ImageCaptioningLstm, self).__init__()
        self.pad_idx = pad_idx

        # CNN encoder
        resnet = models.resnet50(weights='IMAGENET1K_V2')
        for param in resnet.parameters():
            param.requires_grad = False
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last two layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        # Embeddings
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float(), freeze=True)
        
    def forward(self, images, captions):
        # Encode images
        features = self.cnn(images)  # [batch_size, C, H, W]
        features = self.avgpool(features).squeeze(-1).squeeze(-1)  # [batch_size, C]
        features = self.cnn_linear(features)  # [batch_size, embed_size]
        features = self.bn(features)  # [batch_size, embed_size]
        features = features.unsqueeze(1)  # [batch_size, 1, embed_size]
    
        # Prepare captions
        embeddings = self.embedding(captions)  # [batch_size, seq_len, embed_size]
            
        combined_input = torch.cat((features, embeddings), dim=1)  # [batch_size, 1 + seq_len, embed_size]
        # Pass through LSTM
        aggregated_h, (ht, ct) = self.lstm(combined_input)  # LSTM processes combined input
        # Generate output
        
        # Use the last hidden state (ht[-1]) for prediction
        last_hidden_state = ht[-1]  # [batch_size, hidden_size]
    
        output = self.linear(last_hidden_state)
        return output
    
