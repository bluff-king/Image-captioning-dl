from evaluation.BLEU import BLEU 
from evaluation.METEOR import METEOR
from evaluation.ROUGE import ROUGE_L
import json
import pandas as pd 
import numpy as np 

from models.transformer import ImageCaptioningTransformer
import torch
from PIL import Image
from dataset.dataset import transform
from embedding.embedding import vocab_npa, stoi

import re 

with open("config.json", "r") as json_file:
    cfg = json.load(json_file)

CHECKPOINT_PATH = cfg['paths']['checkpoint_path']
CAPTIONS_PATH = cfg["paths"]["captions_path"]
IMAGE_PATH = cfg["paths"]["image_path"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pad_idx = stoi('<PAD>')
sos_idx = stoi('<SOS>')
eos_idx = stoi('<EOS>')
unk_idx = stoi('<UNK>')

CAPTIONS_LENGTH = 20
num_captions = 5
temperature = 0.2
max_unk_wait = 30

references = []

test_data = pd.read_csv(f"{CAPTIONS_PATH}test.txt", header=None)
test_data_unique = test_data.drop_duplicates(subset=[0])

image_ids = list(test_data_unique[0])
test_captions = list(test_data[1])

for i in range(0, len(test_captions), 5):
    references.append(test_captions[i:i+5])


trained_model = ImageCaptioningTransformer(cap_len=CAPTIONS_LENGTH).to(device)
state_dict = torch.load(
    f'{CHECKPOINT_PATH}transformer_caplen{CAPTIONS_LENGTH}.pth',
    weights_only=True
)['model_state_dict']
trained_model.load_state_dict(state_dict)
trained_model.eval()


def generate_caption_single_img(model, img_jpg, num_captions=1):
    img = transform(
        Image.open(f'{IMAGE_PATH}{img_jpg}').convert('RGB')
    )
    img = torch.unsqueeze(img, 0).to(device)

    candidates = []
    for _ in range(num_captions):
        caption = [[pad_idx for _ in range(CAPTIONS_LENGTH)]]
        caption = torch.tensor(caption).to(device)
        caption[0, 0] = sos_idx
        caption_str = '<SOS> '

        unk_wait = 0
        next_idx = 1
        with torch.no_grad():
            while next_idx < CAPTIONS_LENGTH:
                output_logits = model(img, caption)
                output_logits = output_logits / temperature

                output_probs = torch.softmax(output_logits, dim=-1)
                output = torch.multinomial(output_probs, num_samples=1).item()

                if output == unk_idx:
                    unk_wait += 1
                    if unk_wait < max_unk_wait:
                        continue
                
                unk_wait = 0

                caption_str += f'{vocab_npa[output]} '
                caption[0, next_idx] = output

                if output == eos_idx:
                    break

                next_idx += 1
        caption_str = re.sub(r'[<SOS><EOS>]', '', caption_str)
        caption_str = caption_str.strip()
        candidates.append(caption_str)
    return candidates

def compute_scores_random(model, image_ids, references):
    candidates = []
    cnt = 0
    for img_jpg in image_ids:
        candidates.extend(generate_caption_single_img(model, img_jpg))
        cnt += 1
        print(cnt)
        
    # we will calculate BLEU, METEOR, ROUGE-L here 
    bleu = BLEU(references, candidates)
    meteor = METEOR(references, candidates)
    rouge_l = ROUGE_L(references, candidates)
    d = {'BLEU-1': bleu.bleu1(), 'BLEU-4': bleu.bleu4(), 'METEOR': meteor.meteor(), 'ROUGE-L': rouge_l.rouge_l()}

    return list(d.items()), candidates

def compute_scores_best_of_n(model, image_ids, references):
    candidates = []
    cnt = 0
    for img_jpg in image_ids: 
        candidates.append(generate_caption_single_img(model, img_jpg, num_captions=num_captions))
        cnt += 1
        print(cnt)
    
    bleu1 = []
    bleu4 = []
    meteor = []
    rouge_l = []
    for candidate, reference in zip(candidates, references):
        bleu1.append(max(BLEU([reference], [sentence]).bleu1() for sentence in candidate))      
        bleu4.append(max(BLEU([reference], [sentence]).bleu4() for sentence in candidate))
        meteor.append(max(METEOR([reference], [sentence]).meteor() for sentence in candidate))
        rouge_l.append(max(ROUGE_L([reference], [sentence]).rouge_l() for sentence in candidate))  
    
    d = {'BLEU-1': np.mean(bleu1), 'BLEU-4': np.mean(bleu4), 'METEOR': np.mean(meteor), 'ROUGE-L': np.mean(rouge_l)}
    
    return list(d.items())

scores_random = compute_scores_random(model=trained_model, image_ids=image_ids, references=references)[0]
print(scores_random)

scores_best_of_n = compute_scores_best_of_n(model=trained_model, image_ids=image_ids, references=references)
print(scores_best_of_n)






