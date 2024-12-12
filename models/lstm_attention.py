import torch
from torch import nn
import numpy as np
from torchvision import models
from models.attention import Attention
from embedding.embedding import embs_npa, vocab_npa, stoi

import json

with open("config.json", "r") as json_file:
    cfg = json.load(json_file)
    
lstm_params = cfg['hyperparameters']['glove_lstm']
CAPTIONS_LENGTH = lstm_params['captions_length']
HIDDEN_SIZE = lstm_params['hidden_dim']
ATTENTION_DIM = lstm_params['attention_dim']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pad_idx = stoi("<PAD>")


class ImageCaptioningLstm(nn.Module):
    def __init__(self, embed_size = embs_npa.shape[1], vocab_size = len(vocab_npa), hidden_size = HIDDEN_SIZE,pad_idx = pad_idx, attention_dim = ATTENTION_DIM, dropout = 0.3):
        super(ImageCaptioningLstm, self).__init__()
        self.pad_idx = pad_idx

        # CNN encoders
        resnet = models.resnet50(weights='IMAGENET1K_V2')
        for param in resnet.parameters():
            param.requires_grad = False
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last two layers
        self.avgpool = nn.AdaptiveAvgPool2d((14, 14))
        self.cnn_linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
        self.attention = Attention(features_dim = embed_size, decoder_dim = hidden_size, attention_dim = attention_dim, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True, bias = True)
        
        self.linear = nn.Linear(hidden_size + embed_size, vocab_size)
        
        # Embeddings
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float(), freeze=True)
        
    def forward(self, images, captions):
        # Encode images
        features = self.cnn(images)  # [batch_size, C, H, W]
        features = self.avgpool(features)  # [batch_size, 2048, 14,14]

        features = features.flatten(start_dim = 2).permute(0,2,1)
        
        features = self.cnn_linear(features)  # [batch_size,156, embed_size]
        features = self.bn(features.transpose(1,2)).transpose(1,2)  # [batch_size,156, embed_size]
    
        # Prepare captions
        embeddings = self.embedding(captions)  # [batch_size, seq_len, embed_size]

        
            
        combined_input = torch.cat((features, embeddings), dim=1)  # [batch_size, 1 + seq_len, embed_size]
        # Pass through LSTM
        aggregated_h, (ht, ct) = self.lstm(combined_input)  # LSTM processes combined input
        # Generate output
        
        attention_weight_encoding, _ = self.attention(features, ht[-1])

        combined = torch.cat((ht[-1], attention_weight_encoding), dim = 1)
        # Use the last hidden state (ht[-1]) for prediction
    
        output = self.linear(combined)
        return output
    
