from PIL import Image
import os
import torch
from embedding.embedding import stoi, numericalize
from torch.utils.data import Dataset
from torchvision import transforms

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
