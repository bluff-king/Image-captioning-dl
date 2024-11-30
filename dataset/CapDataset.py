import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class CapDataset(Dataset):
    def __init__(
            self,args, tokenizer: PreTrainedTokenizer, mode: str = "train"
    ) -> None:
        super().__init__()

        self.args = args

        self.image_features = torch.load(self.args.features_path)

        captions_df = pd.read_csv(self.args.caption_file, sep=',', header = None, names = ['image', 'caption'])
        captions_df['image'] = captions_df['image'].str.strip()

        self.data = captions_df[captions_df['image'].isin(self.image_features.keys())].reset_index(drop =True)

        self.tokenizer = tokenizer
        self.max_seq_len = self.args.max_seq_len

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self,index:int):
        row = self.data.iloc[index]
        image_name = row['image']
        caption = row['caption']

        image_feature = self.image_features[image_name]

        caption_encoding = self.tokenizer(
            caption,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )

        return{
            'image_feature': torch.tensor(image_feature, dtype = torch.float),
            'input_ids': caption_encoding['input_ids'].squeeze(0),
            'attention_mask': caption_encoding['attention_mask'].squeeze(0),
        }
