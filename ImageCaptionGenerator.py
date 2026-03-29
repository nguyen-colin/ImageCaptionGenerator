from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

images = BASE_DIR / "archive" / "Images"
captions = BASE_DIR / "archive" / "captions.txt"

def load_captions(file_path):
    with open(file_path, 'r') as f:
        captions = f.readlines()
        captions = [caption.lower() for caption in captions[1:]]
    return captions

def load_images(image_path):
    return [f for f in os.listdir(image_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

captions = load_captions(captions)
images = load_images(images)
print(captions[:15:3])
print(images[:15])
print("Captions loaded:", len(captions))
print("Images found:", len(images))
print("Sample captions:", captions[:5])
print("Sample images:", images[:5])