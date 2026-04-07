from pathlib import Path
from transformers import PreTrainedTokenizerFast
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
# print(f"Cleaned Captions: {cleaned_captions[:20:2]}")



def create_caption_ids(captions, cleaned_captions):
    captions_IDS = []
    for i in range(len(cleaned_captions)):
        item = captions[i].split(',')[0]+'\t' + 'start ' + cleaned_captions[i] + ' end\n'
        captions_IDS.append(item)
    return captions_IDS

def visualize_captions(captions_IDS, num_samples=5):
    caption_dict = {}
    for caption in captions_IDS:
        image_id, caption_text = caption.split('\t')
        if image_id not in caption_dict:
            caption_dict[image_id] = []
        caption_dict[image_id].append(caption_text.strip())
    else:
        list_captions = [x for x in caption_dict.items()]
    
    count = 1
    fig = plt.figure(figsize=(10, 5))
    for image_id, caption_text in list_captions[:num_samples]:
        captions = caption_dict[image_id]
        image_load = image.load_img(str(BASE_DIR / "archive" / "Images" / image_id), target_size=(224, 224))
        ax = fig.add_subplot(num_samples, 2, count, xticks=[], yticks=[])
        ax.imshow(image_load)
        count += 1

        ax = fig.add_subplot(num_samples, 2 ,count)
        plt.axis("off")
        ax.plot()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(captions))
        for i, caption in enumerate(captions):
            ax.text(0, i, caption, fontsize=20)
        count += 1
    plt.show()

captions_IDS = create_caption_ids(captions, cleaned_captions)
# print(f"Caption IDs: {captions_IDS[:20]}")

visualize_captions(captions_IDS, num_samples=5)

    

# Function to tokenize captions using a pretrained tokenizer
def tokenize_captions(captions, tokenizer):
    tokenized_captions = []
    for caption in captions:
        tokens = tokenizer.encode(caption)
        tokenized_captions.append(tokens)
    return tokenized_captions

tokens = tokenize_captions(captions, PreTrainedTokenizerFast.from_pretrained('bert-base-uncased'))
# print(f"Tokens: {tokens}")

# Splitting the dataset into training, validation and test sets
# We will use an 80-10-10 split for training, validation and testing respectively
def split_dataset(images, captions_IDS):
    train_caption_id, temp_caption_id = train_test_split(images, test_size=0.2, random_state=42)
    val_captions_id, test_captions_id = train_test_split(temp_caption_id, test_size=0.5, random_state=42)
    train_caption, val_caption, test_caption = [], [], []
    for caption in captions_IDS: 
        image_id, _ = caption.split('\t', 1)
        if image_id in train_caption_id:
            train_caption.append(caption)
        elif image_id in val_captions_id:
            val_caption.append(caption)
        elif image_id in test_captions_id:
            test_caption.append(caption)
    return train_caption, val_caption, test_caption, train_caption_id, val_captions_id, test_captions_id

train_caption, val_caption, test_caption, train_caption_id, val_captions_id, test_captions_id = split_dataset(images, captions_IDS)
# print(f"Total captions: {len(images)}")
print(f"Sample training caption: {train_caption[0]}")
print(f"Sample validation caption: {val_caption[0]}")
print(f"Sample test caption: {test_caption[0]}")

# Image feature extraction using a pretrained model (ResNet50)
# include_top=False removes the classifier
# pooling='avg' gives a single 2048-d feature vector for each image
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
def extract_image_features(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

print(extract_image_features(str(BASE_DIR / "archive" / "Images" / images[0]), resnet_model))

train_image_embeddings, val_image_embeddings, test_image_embeddings = {}, {}, {}
def extract_features_for_dataset(image_path, model, train_captions_id, val_captions_id, test_captions_id, train_image_embeddings, val_image_embeddings, test_image_embeddings):
    pbar = tqdm(total=len(image_path), position=0)
    for image_id in image_path:
        image_embedding = extract_image_features(str(BASE_DIR / "archive" / "Images" / image_id), model)
        if image_id in train_captions_id:
            train_image_embeddings[image_id] = image_embedding.flatten()
        elif image_id in val_captions_id:
            val_image_embeddings[image_id] = image_embedding.flatten()
        elif image_id in test_captions_id:
            test_image_embeddings[image_id] = image_embedding.flatten()
        pbar.update(1)
    pbar.close()
extract_features_for_dataset(images, resnet_model, train_caption_id, val_captions_id, test_captions_id, train_image_embeddings, val_image_embeddings, test_image_embeddings)
print(f"Extracted image features for training set: {len(train_image_embeddings)}")
print(f"Extracted image features for validation set: {len(val_image_embeddings)}")
print(f"Extracted image features for test set: {len(test_image_embeddings)}")
  