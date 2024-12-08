from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.quantization import quantize_dynamic

model_name = "unitary/toxic-bert"

# Using tqdm to show progress for model retrieval
print(f'Retrieving {model_name} from transformers...')
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print('Quantizing model...')
# Here we can simulate a process that takes time, for example, if we had multiple layers to quantize
layers_to_quantize = [nn.Linear]  # Example list of layers to quantize
for layer in tqdm(layers_to_quantize, desc="Quantizing layers"):
    quantized_model = quantize_dynamic(model, {layer}, dtype=torch.qint8)  # Quantize each layer
print('Quantized model created')

print('Saving model')
# Save the model
torch.save(quantized_model, 'toxic_bert.pth')  # Directly save the model object
print('Model saved')
