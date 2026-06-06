import base64
import requests

SERVER = "http://127.0.0.1:8081/v1/chat/completions"

PROMPT = '''
You are creating a dataset for an image analysis model designed to identify potential mosquito breeding sites. The dataset will consist of image-question-answer pairs. 

For the provided image:
1. Generate a question asking whether it depicts a potential mosquito breeding site
2. Provide a detailed answer explaining why or why not

Output ONLY in this exact format (no extra text, no explanations):

Question: Does this image depict a potential mosquito breeding site?

Response: [Yes/No]

Reasoning(Why): [Your detailed reasoning here]
'''

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_image(path, prompt=PROMPT, max_tokens=1024):
    data_uri = f"data:image/jpeg;base64,{encode_image(path)}"
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    r = requests.post(SERVER, json=payload, timeout=300)
    r.raise_for_status()
    
    msg = r.json()["choices"][0]["message"]
    
    # Combine both reasoning_content (thinking) + content (final answer)
    reasoning = msg.get("reasoning_content", "")
    content = msg.get("content", "")
    
    # If content is empty, use reasoning; otherwise combine them
    if content.strip():
        answer = content
    else:
        answer = reasoning
    
    return answer


if __name__ == "__main__":
    print(describe_image("images/frame.jpg"))
