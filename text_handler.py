from typing import Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

from model_definition import get_model
from tokenizer import get_tokenizer


class TextHandler(BaseHandler):
  def initialize(self, context, toxic_threshold:float=0.5):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = get_model()
    self.tokenizer = get_tokenizer()
    self.max_length = 128

    self.toxic_threshold = toxic_threshold

    self.model = self.model.to(self.device)
    self.model.eval()

  def preprocess(self, text):
      # Tokenize the text for Toxic-BERT
      tokenized_inputs = self.tokenizer(
         text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
      inputs = {
          "input_ids": tokenized_inputs["input_ids"].to(torch.device('cpu')),
          "attention_mask": tokenized_inputs["attention_mask"]
        }
      return inputs

  def inference(self, inputs):
    self.model.eval()

    with torch.no_grad():
        outputs = self.model(**inputs)
    return outputs

  def postprocess(self, outputs)->Dict[str, float]:
    with torch.no_grad():
      # Assuming 'outputs' is your SequenceClassifierOutput object
      logits = outputs.logits

      # Apply softmax to get probabilities
      probabilities = torch.softmax(logits, dim=1)

      indices_above_threshold = torch.nonzero(probabilities > self.toxic_threshold)

      # Create the dictionary
      results_dict = {}
      for index in indices_above_threshold:
          label_index = index[1].item()  # Get the class index (column index)
          probability = probabilities[index[0], index[1]].item()  # Get the probability

          label = labels[label_index]  # Get the label from id2label

          results_dict[label] = probability
    return results_dict