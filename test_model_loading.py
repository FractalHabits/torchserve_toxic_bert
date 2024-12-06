import torch

def test_model_loading():
    try:
        model = torch.load('toxic_bert.pth')
        print('Model loaded successfully.')
    except Exception as e:
        print(f'Error loading model: {e}')

if __name__ == "__main__":
    test_model_loading()