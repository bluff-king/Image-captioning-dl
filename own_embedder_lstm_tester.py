from models.own_embedder_lstm import ImageCaptioningLstm
import torch
from PIL import Image
from dataset.dataset import transform

from embedding.embedding import own_vocab_npa, own_stoi

import warnings
warnings.filterwarnings('ignore')


import json
with open("config.json", "r") as json_file:
    cfg = json.load(json_file)


CAPTIONS_LENGTH = cfg['hyperparameters']['own_embedder_lstm']['captions_length']
CHECKPOINT_PATH = cfg['paths']['checkpoint_path']
img_path = cfg['paths']['image_path']

img_jpg = '1322323208_c7ecb742c6.jpg'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pad_idx = own_stoi('<PAD>')
sos_idx = own_stoi('<SOS>')
eos_idx = own_stoi('<EOS>')
unk_idx = own_stoi('<UNK>')

num_captions = 10
temperature = 0.2

def main() -> None:
    trained_model = ImageCaptioningLstm().to(device)
    state_dict = torch.load(
        f'{CHECKPOINT_PATH}own_embedder_lstm{CAPTIONS_LENGTH}.pth',
        weights_only=True
    )['model_state_dict']
    trained_model.load_state_dict(state_dict)
    trained_model.eval()

    img = transform(
        Image.open(f'{img_path}{img_jpg}').convert('RGB')
    )
    img = torch.unsqueeze(img, 0).to(device)

    for _ in range(num_captions):
        caption = [[pad_idx for _ in range(CAPTIONS_LENGTH)]]
        caption = torch.tensor(caption).to(device)
        caption[0, 0] = sos_idx
        caption_str = '<SOS> '

        next_idx = 1
        with torch.no_grad():
            while next_idx < CAPTIONS_LENGTH:
                output_logits = trained_model(img, caption)
                output_logits = output_logits / temperature

                output_probs = torch.softmax(output_logits, dim=-1)
                output = torch.multinomial(output_probs, num_samples=1).item()

                if output == unk_idx:
                    continue

                caption_str += f'{own_vocab_npa[output]} '
                caption[0, next_idx] = output

                if output == eos_idx:
                    break

                next_idx += 1
        print(caption_str)

if __name__ == '__main__':
    main()