import base64
import requests

SERVER = "http://127.0.0.1:8081/v1/chat/completions"


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_image(path, prompt="Describe this image.Keep it to a paragraph.", max_tokens=256):
    data_uri = f"data:image/jpeg;base64,{encode_image(path)}"
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    r = requests.post(SERVER, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(describe_image("images/test-tire-water.jpg"))
