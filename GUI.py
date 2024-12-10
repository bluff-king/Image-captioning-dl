import gradio as gr 
import torch 
from models.transformer import ImageCaptioningTransformer
from embedding.embedding import stoi, vocab_npa
from PIL import Image 
from torchvision import transforms
import re 

import warnings
warnings.filterwarnings('ignore')


import json
with open("config.json", "r") as json_file:
    cfg = json.load(json_file)

# Constants
temperature = 0.2
num_captions = 5
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
}


# This function is used to load all the model by the name of model
def load_model(model_name, checkpoint_path):
    if model_name in ['Transformer20', 'Transformer25', 'Transformer20_COCO', 'Transformer25_COCO']:
        model = ImageCaptioningTransformer().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model 


all_models = {}
for model_name, checkpoint_path in checkpoint_paths.items():
    all_models[model_name] = load_model(model_name, checkpoint_path)


# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 

# This function is used to generate caption for each model 

def generate_caption(image, model_name):
    model = all_models[model_name]
    model.to(device)

    img = transform(Image.fromarray(image).convert("RGB"))
    img = torch.unsqueeze(img, 0).to(device)

    if model_name in ['Transformer20', 'Transformer20_COCO']:
        max_caption_len = 20
    
    elif model_name in ['Transformer25', 'Transformer25_COCO']:
        max_caption_len = 25
    
    else: 
        max_caption_len = 15
        
    ans = []
    for _ in range(num_captions):
        caption = [[pad_idx for _ in range(max_caption_len)]]
        caption = torch.tensor(caption).to(device)
        caption[0, 0] = sos_idx
        caption_str = '<SOS> '

        next_idx = 1
        with torch.no_grad():
            while next_idx < max_caption_len:
                output_logits = model(img, caption)
                output_logits = output_logits / temperature

                output_probs = torch.softmax(output_logits, dim=-1)
                output = torch.multinomial(output_probs, num_samples=1).item()

                if output == unk_idx:
                    continue

                caption_str += f'{vocab_npa[output]} '
                caption[0, next_idx] = output

                if output == eos_idx:
                    break

                next_idx += 1
        ans.append(re.sub(r'[<SOS><EOS>]', '', caption_str))
        
    return ans

# Gradio interface
interface = gr.Interface(
    fn=generate_caption,
    inputs=[
        gr.Image(type="numpy", label="Upload an image"),
        gr.Dropdown(choices=list(all_models.keys()), label="Select Model")
    ],
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Captioning",
    description="Upload an image to generate a caption using the trained model.",
    theme="default"
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()
