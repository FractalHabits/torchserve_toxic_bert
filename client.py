import requests
import argparse

def predict_toxicity(text, threshold=0.0000001):
    url = "http://127.0.0.1:8080/predictions/toxic_bert"
    # Simple text request without auth headers
    try:
        #print(f'Sending post request to {url}')
        response = requests.post(url, data={'text': text, 'threshold': threshold})
        #print(f'Response: {response.json()}')

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response text: {e.response.text}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Text to analyze")
    parser.add_argument("--threshold", type=float, default=0.0000001, help="Toxicity threshold")
    args = parser.parse_args()
    
    #print(f'Sending text: {args.text}, & threshold: {args.threshold} to server...')
    result = predict_toxicity(args.text, args.threshold)
    #print(f'Received result: {result}')
    if result:
        print(f"\nResult:")
        for label, probability in result.items():
            print(f"\n{label}: {probability}")