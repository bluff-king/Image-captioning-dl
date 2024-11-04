from PIL import Image
import torch
import torchvision.transforms as transforms 
import torchvision.models as models 
import torch.nn as nn 
import numpy as np 

class Img2Vec():
    
    def __init__(self, model):
        
        self.OUTPUT_SIZES = {
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
            'resnet152': 2048,
            'inceptionv3': 2048,
            'alexnet': 4096,
            'vgg11bn': 4096,
            'vgg13bn': 4096,
            'vgg16bn': 4096,
            'vgg19bn': 4096
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model
        self.layer_output_size = self.OUTPUT_SIZES[model]
        self.model, self.extraction_layer = self._get_model_and_layer(model)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.model_name == 'inceptionv3':
            self.scaler = transforms.Resize((299, 299))
        elif self.model_name == 'alexnet':
            self.scaler = transforms.Resize((227, 227))
        else:     
            self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # mean and std of ImageNet
        self.to_tensor = transforms.ToTensor()
    
    
    def _get_model_and_layer(self, model_name):
        
        if model_name == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1')
            layer = model._modules.get('avgpool')
            
        elif model_name == 'resnet34':
            model = models.resnet34(weights='IMAGENET1K_V1')
            layer = model._modules.get('avgpool')
            
        elif model_name == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V2')
            layer = model._modules.get('avgpool')
            
        elif model_name == 'resnet101':
            model = models.resnet101(weights='IMAGENET1K_V2')
            layer = model._modules.get('avgpool')
            
        elif model_name == 'resnet152':
            model = models.resnet152(weights='IMAGENET1K_V2')
            layer = model._modules.get('avgpool')
        
        elif model_name == 'inceptionv3':
            model = models.inception_v3(weights='IMAGENET1K_V1')
            layer = model._modules.get('dropout')
        
        elif model_name == 'alexnet':
            model = models.alexnet(weights='IMAGENET1K_V1')
            layer = model.classifier[-2]
        
        elif model_name == 'vgg11bn':
            model = models.vgg11_bn(weights='IMAGENET1K_V1')
            layer = model.classifier[-2]        
        
        elif model_name == 'vgg13bn':
            model = models.vgg13_bn(weights='IMAGENET1K_V1')
            layer = model.classifier[-2]        
        
        elif model_name == 'vgg16bn':
            model = models.vgg16_bn(weights='IMAGENET1K_V1')
            layer = model.classifier[-2]        
        
        elif model_name == 'vgg19bn':
            model = models.vgg19_bn(weights='IMAGENET1K_V1')
            layer = model.classifier[-2]        
        
        
        return model, layer 

    def eval(self):
        self.model.eval()

    def get_vector(self, img_name):
        
        # Load the image with Pillow library
        img = Image.open(img_name).convert('RGB')
        
        # Create a PyTorch Variable with the transformed image
        transformed_img = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)
        
        # Create a vector of zeros that will hold our feature vector 
        if self.model_name in ['alexnet', 'vgg11bn', 'vgg13bn', 'vgg16bn', 'vgg19bn']:
            my_embedding = torch.zeros(1, self.layer_output_size).to(self.device)
        
        else:
            my_embedding = torch.zeros(1, self.layer_output_size, 1, 1).to(self.device) 
        
        # Define a function that will copy the output of a layer 
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)
        
        # Attach that function to our selected layer 
        h = self.extraction_layer.register_forward_hook(copy_data)
        
        # Run the model on our transformed image
        self.model(transformed_img)
        
        # Detach our copy function from the layer
        h.remove()
        
        # Return the feature vector 
        if self.model_name in ['alexnet', 'vgg11bn', 'vgg13bn', 'vgg16bn', 'vgg19bn']:
            # return my_embedding.cpu().numpy().flatten()
            return my_embedding.cpu().flatten()

        else:
            # return my_embedding.cpu().numpy().flatten()
            return my_embedding.cpu().flatten()
