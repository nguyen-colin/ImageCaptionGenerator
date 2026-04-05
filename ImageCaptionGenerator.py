from pathlib import Path
from transformers import PreTrainedTokenizerFast
import os
import re

# Define the base directory and paths to images and captions
BASE_DIR = Path(__file__).resolve().parent
images = BASE_DIR / "archive" / "Images"
captions = BASE_DIR / "archive" / "captions.txt"


# Function to load captions from the text file
def load_captions(file_path):
    with open(file_path, 'r') as f:
        captions = f.readlines()
        captions = [caption.lower().strip() for caption in captions[1:]]
    return captions

# Function to load image file names from the specified directory
def load_images(image_path):
    return [f for f in os.listdir(image_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

captions = load_captions(captions)
images = load_images(images)

# Clean up captions by removing newlines and extra spaces
def clean_captions(captions):
    cleaned_captions = []
    for caption in captions:
        cleaned_caption = caption.split(',')[1]
        cleaned_caption = re.sub(r'[^\w\s]', '', cleaned_caption)
        cleaned_caption = re.sub(r'\d+', '', cleaned_caption)
        cleaned_caption = re.sub(r'\s+', ' ', cleaned_caption).strip()
        cleaned_captions.append(cleaned_caption)
    return cleaned_captions
        
cleaned_captions = clean_captions(captions)
# print(cleaned_captions[:20:2])

#def create_image_caption_pairs(images, cleaned_captions):

def create_caption_ids(captions, cleaned_captions):
    captions_IDS = []
    for i in range(len(cleaned_captions)):
        item = captions[i].split(',')[0]+'\t' + 'start ' + cleaned_captions[i] + ' end\n'
        captions_IDS.append(item)
    return captions_IDS

captions_IDS = create_caption_ids(captions, cleaned_captions)
# print(captions_IDS[:20:3])

# Function to tokenize captions using a pretrained tokenizer
def tokenize_captions(captions, tokenizer):
    tokenized_captions = []
    for caption in captions:
        tokens = tokenizer.encode(caption)
        tokenized_captions.append(tokens)
    return tokenized_captions

tokens = tokenize_captions(captions, PreTrainedTokenizerFast.from_pretrained('bert-base-uncased'))
# print(tokens)