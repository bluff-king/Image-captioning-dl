from dataset.own_embedder_data import get_mapped_data
from dataset.word_index_dict import description_tokens, clean_and_tokenize

all_mapping = get_mapped_data()
word_to_index_dict, index_to_word_dict = description_tokens(all_mapping)

def NumericalizedDescriptions(all_mapping = all_mapping, word_to_index_dict = word_to_index_dict):
    all_description_indices = []
    
    for _, captions in all_mapping.items():
        for caption in captions:
            tokens = clean_and_tokenize(caption)
            indices = [word_to_index_dict[token] for token in tokens]
            all_description_indices.append(indices)
    
    return all_description_indices