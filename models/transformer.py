import torch
from torch import nn
import numpy as np
from torchvision import models
from embedding.embedding import embs_npa, vocab_npa, stoi

import json

with open("config.json", "r") as json_file:
    cfg = json.load(json_file)

transformer_params = cfg['hyperparameters']['transformer']
CAPTIONS_LENGTH = cfg['hyperparameters']['captions_length']
NUM_HEADS = transformer_params['num_heads']
NUM_ENCODER_LAYERS = transformer_params['num_encoder_layers']
NUM_DECODER_LAYERS = transformer_params['num_decoder_layers']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pad_idx = stoi("<PAD>")


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, maxlen=50):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)
                        * (np.log(10000) / emb_size))
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.pos_embedding = pos_embedding.unsqueeze(-2).to(device)

    def forward(self, x):
        x = x + self.pos_embedding[:x.size(0), :]
        return x


class ImageCaptioningTransformer(nn.Module):
    def __init__(
        self,
        embed_size=embs_npa.shape[1],
        vocab_size=len(vocab_npa),
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        pad_idx=pad_idx
    ):
        super(ImageCaptioningTransformer, self).__init__()
        self.pad_idx = pad_idx

        # CNN Encoder
        resnet = models.resnet50(weights='IMAGENET1K_V2')
        for param in resnet.parameters():
            param.requires_grad = False
        # Remove the last two layers
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=num_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Embeddings
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(embs_npa).float(), freeze=True)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(emb_size=embed_size)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, nhead=num_heads
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Output Layer
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def make_src_mask(self, src):
        # src: [batch_size, src_len]
        src_mask = (src != self.pad_idx).unsqueeze(-2)
        return src_mask

    def forward(self, images, captions):
        # Encode images
        features = self.cnn(images)  # [batch_size, C, H, W]
        features = self.avgpool(
            features).squeeze(-1).squeeze(-1)  # [batch_size, C]
        features = self.cnn_linear(features)  # [batch_size, embed_size]
        features = self.bn(features)  # [batch_size, embed_size]
        features = features.unsqueeze(1)  # [batch_size, 1, embed_size]

        # Transformer Encoder
        encoder_outputs = self.transformer_encoder(
            features.permute(1, 0, 2))  # [1, batch_size, embed_size]

        # Prepare captions
        # [batch_size, seq_len, embed_size]
        embeddings = self.embedding(captions)
        # [seq_len, batch_size, embed_size]
        embeddings = embeddings.permute(1, 0, 2)
        embeddings = self.positional_encoding(embeddings)

        # Create masks
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(CAPTIONS_LENGTH).to(embeddings.device)
        tgt_mask = torch.triu(torch.ones(
            (CAPTIONS_LENGTH, CAPTIONS_LENGTH), device=captions.device), diagonal=1).bool()

        tgt_key_padding_mask = (captions == self.pad_idx)  # Corrected line

        # Transformer Decoder
        outputs = self.transformer_decoder(
            embeddings,
            encoder_outputs,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [seq_len, batch_size, embed_size]

        # Extract the last time step's output
        outputs = outputs[-1, :, :]  # [batch_size, embed_size]

        outputs = self.fc_out(outputs)  # [batch_size, vocab_size]

        return outputs  # [batch_size, vocab_size]
