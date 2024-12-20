import gradio as gr 
import torch 
from models.transformer import ImageCaptioningTransformer
from models.lstm import ImageCaptioningLstm as LstmGlove
from models.lstm_attention import ImageCaptioningLstm as LstmAtt
from models.own_embedder_lstm import ImageCaptioningLstm as LstmCBOW
from embedding.embedding import stoi, vocab_npa
from PIL import Image 
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer
from dataset.dataset import transform
from embedding.own_embedding import own_vocab_npa, own_stoi

import warnings
warnings.filterwarnings('ignore')


import json
with open("config.json", "r") as json_file:
    cfg = json.load(json_file)

# Constants
# temperature = 0.2
# num_captions = 5
pad_idx = stoi('<PAD>')
sos_idx = stoi('<SOS>')
eos_idx = stoi('<EOS>')
unk_idx = stoi('<UNK>')

CHECKPOINT_PATH = cfg['paths']['checkpoint_path']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# All checkpoint_paths stored here 
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


# This function is used to load all the model by the name of model
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
load_success = []
load_fail = []
for model_name, checkpoint_path in checkpoint_paths.items():
    try:
        all_models[model_name] = load_model(model_name, checkpoint_path)
        load_success.append(model_name)
    except:
        load_fail.append(model_name)

warning_str = ''
if load_fail:
    warning_str += f' (Fail to load model(s): {", ".join(load_fail)})'


# This function is used to generate caption for each model
def generate_caption(image, model_name, temperature, num_captions, max_unk_wait):
    # model = all_models[model_name]

    if temperature == 0:
        temperature = 0.01

    img = transform(Image.fromarray(image).convert("RGB"))
    img = torch.unsqueeze(img, 0).to(device)

    if model_name in ['Transformer20', 'Transformer20_COCO', 'LSTM20_Attention']:
        max_caption_len = 20
    
    elif model_name in ['Transformer25', 'Transformer25_COCO', 'LSTM25_Glove', 'LSTM25_CBOW']:
        max_caption_len = 25
    
    # else: 
    #     max_caption_len = 15

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
                    output = torch.multinomial(output_probs, num_samples=1).item()

                    if output == own_unk_idx:
                        unk_wait += 1
                        if unk_wait < max_unk_wait:
                            continue
                    
                    unk_wait = 0

                    # caption_str += f'{vocab_npa[output]} '
                    caption_str.append(own_vocab_npa[output])
                    caption[0, next_idx] = output

                    if output == own_eos_idx:
                        break

                    next_idx += 1

            caption_detokenized = d.detokenize(caption_str)
            ans.append(re.sub(r'[<SOS><EOS>]', '', caption_detokenized))
            
        return "\n".join(ans)

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

                # caption_str += f'{vocab_npa[output]} '
                caption_str.append(vocab_npa[output])
                caption[0, next_idx] = output

                if output == eos_idx:
                    break

                next_idx += 1

        caption_detokenized = d.detokenize(caption_str)
        ans.append(re.sub(r'[<SOS><EOS>]', '', caption_detokenized))
        
    return "\n".join(ans)

# Gradio interface
interface = gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Image(type='numpy', label='Upload an image'),
        gr.Dropdown(choices=load_success, label=f'Select model{warning_str}'),
        gr.Slider(0, 1.0, value=0.2, step=0.05, label='Temperature'),
        gr.Slider(1, 20, value=5, step=1, label='Number of captions'),
        gr.Slider(0, 50, value=20, step=5, label='Max <UNK> wait')
    ],
    outputs=gr.Textbox(label='Generated caption(s)'),
    title='Image Captioning',
    description='Upload an image to generate caption(s) using the trained model.',
    theme='default'
)


# Launch the interface
if __name__ == "__main__":
    interface.launch()
