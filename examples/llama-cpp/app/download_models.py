import os
import requests

models = [
    "https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/resolve/main/codellama-7b.Q5_K_S.gguf",
    "https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_K_M.gguf",
    "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf"
]
for url in models:
    dest = f"models/{url.split('/')[-1]}"
    if os.path.exists(dest):
        print(f"Skipping {dest}, already exists")
        continue
    print(f"Downloading {url} to {dest}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest, 'wb') as f:
            total_downloaded = 0
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    total_downloaded += len(chunk)
                    if total_downloaded >= 10485760:  # 10 MB
                        print('.', end='', flush=True)
                        total_downloaded = 0
        print("\nDownload complete.")
    