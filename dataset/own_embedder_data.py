import tqdm
import os
import json
with open("config.json", "r") as json_file:
    cfg = json.load(json_file)

CAPTIONS_PATH = cfg['paths']['captions_path']

train_filepath = f'{CAPTIONS_PATH}train.txt'
val_filepath = f'{CAPTIONS_PATH}val.txt'

def mapping(file_path):
    ''' Create a dictionary to map 1 image with multiple caps ''' 
    with open(file_path, 'r') as f:
        next(f)
        captions_doc = f.read()
    
    mapping = {}
    for line in tqdm.tqdm(captions_doc.split('\n')):
        img_and_cap = line.split(',')
        # if len(line) < 2:
        #     continue
        image_id, caption = img_and_cap[0], img_and_cap[1:]
        # convert caption list to string
        caption = " ".join(caption)
        if image_id not in mapping:
            mapping[image_id] = []
        # store the caption
        mapping[image_id].append(caption)
    return mapping

def get_mapped_data(train_filepath = train_filepath, val_filepath = val_filepath):
    train_mapping = mapping(train_filepath)
    val_mapping = mapping(val_filepath)
    all_mapping = train_mapping | val_mapping
    return all_mapping



