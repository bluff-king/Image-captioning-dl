import torch
from torch import nn
from dataset.word_index_dict import description_tokens


word_to_index_dict, index_to_word_dict = description_tokens()

class CenterWordPredictor(nn.Module):
    def __init__(self, vocabulary_size, embedding_dimension):    # vocab size = len(word_to_index_dict)
        super(CenterWordPredictor, self).__init__()
        # create an Embedding Layer
        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_dimension)
        # create a Linear layer
        self.decoderLinear = torch.nn.Linear(embedding_dimension, vocabulary_size)           #?
        self.embedding_dim = embedding_dimension
        self.embedding_dimension = embedding_dimension
    def forward(self, contextTsr):
        '''
        processes the context tensor to compute the logits for each word in the vocabulary, 
        representing the model's prediction of the center word based on the provided context.
        '''
        if torch.any(contextTsr >= len(self.embedding.weight)):
            raise ValueError("Index out of bounds in context tensor")
        embedding = self.embedding(contextTsr)  # (batch_size, context_length, embedding_dimension)
        # Average over context words: (batch_size, context_length, embedding_dimension) 
        embedding = torch.mean(embedding, dim=1)
        # -> (batch_size, embedding_dimension): a reprenstative vector for context_tensor
        
        # Decoding, weight, bias in this layer will be random, loss function and backpropagate will fix them later
        outputTsr = self.decoderLinear(embedding)   # logit of each word, size (N, vocab_size)
        return outputTsr
