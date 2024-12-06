import torch
from transformers import AutoModelForSequenceClassification
from torch.quantization import quantize_dynamic
import torch.nn as nn

def get_model(quantize=True):
    print('Loading model...')
    model = torch.load('toxic_bert.pth')
    print('Model loaded')

    if quantize:
        print('Quantizing model...')
        # Assuming you want to quantize all Linear layers
        layers_to_quantize = [nn.Linear]  # Adjust this list as necessary
        for layer in layers_to_quantize:
            model = quantize_dynamic(model, {layer}, dtype=torch.qint8)  # Quantize each layer
        print('Model quantized')

    return model