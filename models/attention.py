import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, features_dim, decoder_dim, attention_dim, dropout=0.5):
        super(Attention, self).__init__()
        self.features_att = nn.utils.weight_norm(nn.Linear(features_dim, attention_dim))  # Linear layer to transform image features
        self.decoder_att = nn.utils.weight_norm(nn.Linear(decoder_dim, attention_dim))  # Linear layer to transform decoder's output
        self.full_att = nn.utils.weight_norm(nn.Linear(attention_dim, 1))  # Linear layer to calculate attention weights
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # Softmax to calculate attention weights

    def forward(self, image_features, decoder_hidden):
        # Compute attention weights
        att1 = self.features_att(image_features)  # (batch_size, 49, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(2)  # (batch_size, 49)
        alpha = self.softmax(att)  # (batch_size, 49)
        attention_weighted_encoding = (image_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, features_dim)
        return attention_weighted_encoding, alpha