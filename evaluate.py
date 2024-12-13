from datetime import datetime
from evaluation.BLEU import BLEU
from evaluation.METEOR import METEOR
from evaluation.ROUGE import ROUGE_L
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

from models.transformer import ImageCaptioningTransformer
import torch
from PIL import Image
from dataset.dataset import transform
from embedding.embedding import vocab_npa, stoi

from nltk.tokenize.treebank import TreebankWordDetokenizer
from models.lstm import ImageCaptioningLstm as LstmGlove
from models.lstm_attention import ImageCaptioningLstm as LstmAtt
from models.own_embedder_lstm import ImageCaptioningLstm as LstmCBOW
from embedding.own_embedding import own_vocab_npa, own_stoi
import re

import warnings
warnings.filterwarnings('ignore')

with open("config.json", "r") as json_file:
    cfg = json.load(json_file)

CHECKPOINT_PATH = cfg['paths']['checkpoint_path']
CAPTIONS_PATH = cfg["paths"]["captions_path"]
IMAGE_PATH = cfg["paths"]["image_path"]

checkpoint_paths = {
    "Transformer20": f"{CHECKPOINT_PATH}transformer_caplen20.pth",
    "Transformer25": f"{CHECKPOINT_PATH}transformer_caplen25.pth",
    "Transformer20_COCO": f"{CHECKPOINT_PATH}transformer_caplen20_coco.pth",
    "Transformer25_COCO": f"{CHECKPOINT_PATH}transformer_caplen25_coco.pth",
    'LSTM25_Glove': f'{CHECKPOINT_PATH}glove_lstm25.pth',
    'LSTM25_CBOW': f'{CHECKPOINT_PATH}own_embedder_lstm25.pth',
    'LSTM20_Attention': f'{CHECKPOINT_PATH}lstm_attention20.pth'
}

d = TreebankWordDetokenizer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pad_idx = stoi('<PAD>')
sos_idx = stoi('<SOS>')
eos_idx = stoi('<EOS>')
unk_idx = stoi('<UNK>')

# CAPTIONS_LENGTH = 20
num_captions = 5
temperature = 0.2
max_unk_wait = 20

references = []

test_data = pd.read_csv(f"{CAPTIONS_PATH}test.txt", header=None)
test_data_unique = test_data.drop_duplicates(subset=[0])

image_ids = list(test_data_unique[0])
test_captions = list(test_data[1])

for i in range(0, len(test_captions), 5):
    references.append(test_captions[i:i+5])

# testing
# image_ids = image_ids[:10]
# references = references[:10]


def load_model(model_name, checkpoint_path):
    model = None

    if model_name in ['Transformer20', 'Transformer20_COCO']:
        model = ImageCaptioningTransformer(cap_len=20).to(device)

    if model_name in ['Transformer25', 'Transformer25_COCO']:
        model = ImageCaptioningTransformer(cap_len=25).to(device)

    if model_name == 'LSTM25_Glove':
        model = LstmGlove().to(device)

    if model_name == 'LSTM25_CBOW':
        model = LstmCBOW().to(device)

    if model_name == 'LSTM20_Attention':
        model = LstmAtt().to(device)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()

        return model

    assert model != None, 'what'

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


all_models = {}
for model_name, checkpoint_path in checkpoint_paths.items():
    all_models[model_name] = load_model(model_name, checkpoint_path)


def generate_caption(image, model_name, temperature=temperature, num_captions=1, max_unk_wait=max_unk_wait):

    if temperature == 0:
        temperature = 0.01

    img = transform(
        Image.open(f'{IMAGE_PATH}{image}').convert('RGB')
    )
    img = torch.unsqueeze(img, 0).to(device)

    if model_name in ['Transformer20', 'Transformer20_COCO', 'LSTM20_Attention']:
        max_caption_len = 20

    elif model_name in ['Transformer25', 'Transformer25_COCO', 'LSTM25_Glove', 'LSTM25_CBOW']:
        max_caption_len = 25

    ans = []

    if model_name == 'LSTM25_CBOW':
        own_pad_idx = own_stoi('<PAD>')
        own_sos_idx = own_stoi('<SOS>')
        own_eos_idx = own_stoi('<EOS>')
        own_unk_idx = own_stoi('<UNK>')
        for _ in range(num_captions):
            caption = [[own_pad_idx for _ in range(max_caption_len)]]
            caption = torch.tensor(caption).to(device)
            caption[0, 0] = own_sos_idx
            caption_str = ['<SOS>']

            unk_wait = 0
            next_idx = 1

            with torch.no_grad():
                while next_idx < max_caption_len:
                    output_logits = all_models[model_name](img, caption)
                    output_logits = output_logits / temperature

                    output_probs = torch.softmax(output_logits, dim=-1)
                    output = torch.multinomial(
                        output_probs, num_samples=1).item()

                    if output == own_unk_idx:
                        unk_wait += 1
                        if unk_wait < max_unk_wait:
                            continue

                    unk_wait = 0
                    caption_str.append(own_vocab_npa[output])
                    caption[0, next_idx] = output

                    if output == own_eos_idx:
                        break

                    next_idx += 1

            caption_detokenized = d.detokenize(caption_str)
            ans.append(re.sub(r'[<SOS><EOS>]', '',
                       caption_detokenized).strip())

        return ans

    for _ in range(num_captions):
        caption = [[pad_idx for _ in range(max_caption_len)]]
        caption = torch.tensor(caption).to(device)
        caption[0, 0] = sos_idx
        caption_str = ['<SOS>']

        unk_wait = 0
        next_idx = 1

        with torch.no_grad():
            while next_idx < max_caption_len:
                output_logits = all_models[model_name](img, caption)
                output_logits = output_logits / temperature

                output_probs = torch.softmax(output_logits, dim=-1)
                output = torch.multinomial(output_probs, num_samples=1).item()

                if output == unk_idx:
                    unk_wait += 1
                    if unk_wait < max_unk_wait:
                        continue

                unk_wait = 0
                caption_str.append(vocab_npa[output])
                caption[0, next_idx] = output

                if output == eos_idx:
                    break

                next_idx += 1

        caption_detokenized = d.detokenize(caption_str)
        ans.append(re.sub(r'[<SOS><EOS>]', '', caption_detokenized).strip())

    return ans


def compute_scores_random(model_name, image_ids, references):
    candidates = []
    for img_jpg in tqdm(image_ids):
        candidates.extend(generate_caption(img_jpg, model_name))

    # we will calculate BLEU, METEOR, ROUGE-L here
    bleu = BLEU(references, candidates)
    meteor = METEOR(references, candidates)
    rouge_l = ROUGE_L(references, candidates)
    d = {'BLEU-1': bleu.bleu1(), 'BLEU-4': bleu.bleu4(),
         'METEOR': meteor.meteor(), 'ROUGE-L': rouge_l.rouge_l()}

    return list(d.items())


def compute_scores_best_of_n(model_name, image_ids, references):
    candidates = []
    for img_jpg in tqdm(image_ids):
        candidates.append(
            generate_caption(img_jpg, model_name, num_captions=num_captions)
        )

    bleu1 = []
    bleu4 = []
    meteor = []
    rouge_l = []
    for candidate, reference in zip(candidates, references):
        bleu1.append(max(BLEU([reference], [sentence]).bleu1()
                     for sentence in candidate))
        bleu4.append(max(BLEU([reference], [sentence]).bleu4()
                     for sentence in candidate))
        meteor.append(max(METEOR([reference], [sentence]).meteor()
                      for sentence in candidate))
        rouge_l.append(
            max(ROUGE_L([reference], [sentence]).rouge_l() for sentence in candidate))

    d = {'BLEU-1': np.mean(bleu1), 'BLEU-4': np.mean(bleu4),
         'METEOR': np.mean(meteor), 'ROUGE-L': np.mean(rouge_l)}

    return list(d.items())


with open('eval_result.txt', 'a') as fw:
    fw.write(f'Time: {str(datetime.now())}\n')
    for model_name in list(all_models.keys()):
        print(f'Evaluating model {model_name}...')
        fw.write(f'{model_name}:\n')
        scores_random = compute_scores_random(
            model_name, image_ids, references)
        fw.write(f'Random: {scores_random}\n')
        scores_best_of_n = compute_scores_best_of_n(
            model_name, image_ids, references)
        fw.write(f'Bo{num_captions}:    {scores_best_of_n}\n\n')
    fw.write('\n')
