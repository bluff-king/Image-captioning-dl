from PIL import Image
import os
import torch
from embedding.embedding import stoi, numericalize
from embedding.embedding import own_stoi, own_numericalize
from torch.utils.data import Dataset
from torchvision import transforms
import random

import json

with open("config.json", "r") as json_file:
    cfg = json.load(json_file)

CAPTIONS_LENGTH = cfg['hyperparameters']['captions_length']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class TransformerDataset(Dataset):
    def __init__(self, root_dir, captions, image_ids, transform=transform):
        self.root_dir = root_dir
        self.transform = transform

        self.captions_augmented = []
        self.imgs_augmented = []
        self.next_token = []
        for caption, img in zip(captions, image_ids):
            numericalized_caption = [stoi("<SOS>")]
            numericalized_caption += numericalize(caption)
            numericalized_caption.append(stoi("<EOS>"))
            for idx in range(min(len(numericalized_caption), CAPTIONS_LENGTH) - 1):
                self.imgs_augmented.append(img)
                # pre-pad here
                padded_caption = numericalized_caption[:idx + 1] \
                    + [stoi('<PAD>')] * CAPTIONS_LENGTH
                padded_caption = padded_caption[:CAPTIONS_LENGTH]
                self.captions_augmented.append(padded_caption)
                self.next_token.append(numericalized_caption[idx + 1])

    def __len__(self):
        return len(self.captions_augmented)

    def __getitem__(self, idx):
        caption = torch.tensor(self.captions_augmented[idx])

        img_id = self.imgs_augmented[idx]
        img_path = os.path.join(self.root_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, caption, self.next_token[idx]


def transformer_collate(batch):
    images = [item[0].unsqueeze(0) for item in batch]
    images = torch.cat(images, dim=0)

    captions = [item[1].unsqueeze(0) for item in batch]
    captions = torch.cat(captions, dim=0)

    next_tokens = torch.tensor([item[2] for item in batch])
    return images, captions, next_tokens


class GloveLstmDataset(Dataset):
    def __init__(self, root_dir, captions, image_ids, transform=transform):
        self.root_dir = root_dir
        self.transform = transform

        self.captions_augmented = []
        self.imgs_augmented = []
        self.next_token = []
        for caption, img in zip(captions, image_ids):
            numericalized_caption = [stoi("<SOS>")]
            numericalized_caption += numericalize(caption)
            numericalized_caption.append(stoi("<EOS>"))
            for idx in range(min(len(numericalized_caption), CAPTIONS_LENGTH) - 1):
                self.imgs_augmented.append(img)
                # pre-pad here
                padded_caption = numericalized_caption[:idx+1] + [stoi('<PAD>')] * CAPTIONS_LENGTH
                padded_caption = padded_caption[:CAPTIONS_LENGTH]
                self.captions_augmented.append(padded_caption)
                self.next_token.append(numericalized_caption[idx + 1])
            
    def __len__(self):
        return len(self.captions_augmented)

    def __getitem__(self, idx):
        caption = torch.tensor(self.captions_augmented[idx])

        img_id = self.imgs_augmented[idx]
        img_path = os.path.join(self.root_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # next_token = itov_single(self.next_token[idx])

        # return image, caption, next_token
        return image, caption, self.next_token[idx]

def glove_lstm_collate(batch):
    images = [item[0].unsqueeze(0) for item in batch]
    images = torch.cat(images, dim=0)

    captions = [item[1].unsqueeze(0) for item in batch]
    captions = torch.cat(captions, dim=0)

    next_tokens = torch.tensor([item[2] for item in batch])
    return images, captions, next_tokens


class OwnLstmDataset(Dataset):
    def __init__(self, root_dir, captions, image_ids, transform=transform):
        self.root_dir = root_dir
        self.transform = transform

        self.captions_augmented = []
        self.imgs_augmented = []
        self.next_token = []
        for caption, img in zip(captions, image_ids):
            numericalized_caption = [own_stoi("<SOS>")]
            numericalized_caption += own_numericalize(caption)
            numericalized_caption.append(own_stoi("<EOS>"))
            for idx in range(min(len(numericalized_caption), CAPTIONS_LENGTH) - 1):
                self.imgs_augmented.append(img)
                # pre-pad here
                padded_caption = numericalized_caption[:idx+1] + [own_stoi('<PAD>')] * CAPTIONS_LENGTH
                padded_caption = padded_caption[:CAPTIONS_LENGTH]
                self.captions_augmented.append(padded_caption)
                self.next_token.append(numericalized_caption[idx + 1])
            
    def __len__(self):
        return len(self.captions_augmented)

    def __getitem__(self, idx):
        caption = torch.tensor(self.captions_augmented[idx])

        img_id = self.imgs_augmented[idx]
        img_path = os.path.join(self.root_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # next_token = itov_single(self.next_token[idx])

        # return image, caption, next_token
        return image, caption, self.next_token[idx]



class ContextToWordDataset(Dataset):
    def __init__(self,
                 all_description_indices,
                 index_to_word_dict,
                 word_to_index_dict,
                 contextLength):
        self.all_description_indices = all_description_indices
        self.index_to_word_dict = index_to_word_dict
        self.word_to_index_dict = word_to_index_dict
        self.contextLength = contextLength

    def __len__(self):
        return len(self.all_description_indices)
    
    def __getitem__(self, idx):
        ''' The method below will return a center word, and several context words around (in indices)'''
        description_indices = self.all_description_indices[idx]        # get cap but presentation is full of indices from created vocab
        
        if len(description_indices) == 0:
            return self.__getitem__((idx + 1) % len(self.all_description_indices))
        
        last_acceptable_center_index = len(description_indices) - 1          # prevent the center word is out of bound
        
        for position, index in enumerate(description_indices):
            last_acceptable_center_index = position
    
        target_idx = random.choice(range(last_acceptable_center_index+1))

        
        context_around_idxs_Tsr = torch.zeros((2 * self.contextLength), dtype=torch.long)
        
        running_idx = target_idx - int(self.contextLength)    # maybe negative      
        counter = 0
        
        # consider all word around target
        while counter < 2 * self.contextLength:
            if running_idx != target_idx:
                # prevent index is negative or out of bound
                if running_idx >= 0 and running_idx < len(description_indices):
                    context_around_idxs_Tsr[counter] = description_indices[running_idx]
                counter += 1
            running_idx += 1
        target_center_word_idx = description_indices[target_idx]
        if target_center_word_idx >= len(self.word_to_index_dict):
            raise ValueError(f"Invalid target index: {target_center_word_idx}")
        return context_around_idxs_Tsr,  target_center_word_idx
