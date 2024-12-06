from transformers import AutoTokenizer

def get_tokenizer():
  model_name = "unitary/toxic-bert"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  return tokenizer