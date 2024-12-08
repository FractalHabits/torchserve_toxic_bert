from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.quantization import quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

class TextHandler(BaseHandler):
    def initialize(self, context, toxic_threshold:float=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = context.system_properties.get("model_dir")
    
        self.tokenizer = self.get_tokenizer()
        self.max_length = 128
        self.model = self.get_model()
    
        self.labels = self.model.config.id2label

        self.toxic_threshold = toxic_threshold
    
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_model(self, quantize=True):
    
        model_name = "unitary/toxic-bert"
    
        # Using tqdm to show progress for model retrieval
        print(f'Retrieving {model_name} from transformers...')
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
        if quantize:
            print('Quantizing model...')
            # Assuming you want to quantize all Linear layers
            layers_to_quantize = [nn.Linear]  # Adjust this list as necessary
            for layer in layers_to_quantize:
                model = quantize_dynamic(model, {layer}, dtype=torch.qint8)  # Quantize each layer
            print('Model quantized')
    
        print('Saving model')
        # Save the model
        torch.save(model, 'toxic_bert.pth')  # Directly save the model object
        print('Model saved')
    
        print('Loading model...')
        model = torch.load(self.model_dir + '/toxic_bert.pth')
        print('Model loaded')
    
        return model

    def get_tokenizer(self):
        model_name = "unitary/toxic-bert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
  
    #def preprocess(self, text, threshold:float=0.0000001):
        
    def preprocess(self, requests):
        """
        requests: List containing a dictionary with text and threshold
        """
        try:
            print("DEBUG: Received requests:", requests)  # Debug print
            
            if len(requests) > 0:
                # Get text and threshold directly from the request
                request = requests[0]
                
                # Convert bytearray to string if necessary
                text = request.get('text', '')
                if isinstance(text, (bytes, bytearray)):
                    text = text.decode('utf-8')
                
                # Get and convert threshold
                threshold = request.get('threshold', '0.0000001')
                if isinstance(threshold, (bytes, bytearray)):
                    threshold = threshold.decode('utf-8')
                self.toxic_threshold = float(threshold)
                
                print(f"DEBUG: Processing text: {text}, threshold: {self.toxic_threshold}")
                
                # Tokenize the text for Toxic-BERT
                tokenized_inputs = self.tokenizer(
                    text, 
                    truncation=True, 
                    padding="max_length", 
                    max_length=self.max_length, 
                    return_tensors="pt"
                )
                
                inputs = {
                    "input_ids": tokenized_inputs["input_ids"].to(self.device),
                    "attention_mask": tokenized_inputs["attention_mask"].to(self.device)
                }
                return inputs
                
        except Exception as e:
            print(f"ERROR in preprocess: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise e

    def inference(self, inputs):
        try:
            print("DEBUG: Starting inference with inputs:", inputs)  # Debug print
            self.model.eval()
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                print("DEBUG: Model output:", outputs)  # Debug print
            return outputs
        except Exception as e:
            print(f"ERROR in inference: {str(e)}")
            import traceback
            print(traceback.format_exc())  # Print full stack trace
            raise e

    def postprocess(self, outputs)->List:
        try:
            print("DEBUG: Starting postprocess with outputs:", outputs)  # Debug print
            with torch.no_grad():
                logits = outputs.logits
                print("DEBUG: Logits shape:", logits.shape)  # Debug print
                
                # Apply softmax to get probabilities
                probabilities = torch.softmax(logits, dim=1)
                print("DEBUG: Probabilities:", probabilities)  # Debug print
                
                indices_above_threshold = torch.nonzero(probabilities > self.toxic_threshold)
                print("DEBUG: Indices above threshold:", indices_above_threshold)  # Debug print
                
                # Create the dictionary with rounded probabilities
                results = {}
                for index in indices_above_threshold:
                    label_index = index[1].item()
                    probability = round(probabilities[0, label_index].item(), 3)  # Round to 3 decimal places
                    label = self.labels[label_index]
                    results[label] = probability
                
                print("DEBUG: Final results:", results)  # Debug print
                # Convert to list format that TorchServe expects
                return [results]
        except Exception as e:
            print(f"ERROR in postprocess: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise e