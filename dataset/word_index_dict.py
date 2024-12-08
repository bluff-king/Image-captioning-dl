from dataset.own_embedder_data import get_mapped_data
import re
all_mapping = get_mapped_data()

def clean_and_tokenize(cap):
    cap = cap.strip('"')
    cap = cap.split()
    # convert to lower case
    cap = [word.lower() for word in cap]
    cap = [word for word in cap if re.match(r'^[a-zA-Z-]+$', word)]
    cap = ' '.join([word for word in cap if len(word) >0])
    cap = cap.split()
    return cap

def description_tokens(mapping = all_mapping):
    number_of_train_lines = 0
    train_token_to_occurrences_dict = {}
    
    for image_id, captions in mapping.items():
        for caption in captions:
            tokens = clean_and_tokenize(caption)
            # Count the occurrences of each token
            for token in tokens:
                if token in train_token_to_occurrences_dict:
                    train_token_to_occurrences_dict[token] += 1
                else:
                    train_token_to_occurrences_dict[token] = 1
            
            number_of_train_lines += 1

    sorted_tokens = sorted(train_token_to_occurrences_dict.items(), key=lambda x: x[1], reverse=True)

    vocab_list = [{"index": idx, "word": token[0], "frequency": token[1]} for idx, token in enumerate(sorted_tokens)]

    word_to_index_dict = {entry["word"]: entry["index"] for entry in vocab_list}
    index_to_word_dict = {entry["index"]: entry["word"] for entry in vocab_list}
    
    return word_to_index_dict, index_to_word_dict

