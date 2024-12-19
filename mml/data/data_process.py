import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer
import pickle
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

class ImageEncoder(nn.Module):
    """
    Encodes image and returns it's embedding.
    """

    def __init__(self, model, device="cpu"):
        super(ImageEncoder, self).__init__()

        self.device = device

        self.preprocessor = CLIPProcessor.from_pretrained(model)
        self.model = CLIPModel.from_pretrained(model).vision_model.to(self.device)

    def forward(self, image):
        # only one image at a time
        image = self.preprocessor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model(**image)

        return image_features.pooler_output

def extract_num(img_name):
    return int(img_name.split("_")[-1].split(".")[0])

def proc_data(clip_model, split, root_dir="coco"):
    model = ImageEncoder(clip_model, 'cuda')
    model.eval()

    embeddings = {}

    img_dir = os.path.join(root_dir, f"{split}2014")
    for img_name in tqdm(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        img = img.convert("RGB")

        img_num = extract_num(img_name)

        with torch.no_grad():
            img_embedding = model(img)

        img_embedding = img_embedding.cpu().numpy().squeeze()

        embeddings[img_num] = img_embedding

    clip_model = clip_model.replace('/', '_')
    output_file = f"{split}_embeddings_{clip_model}.pkl"
    with open(os.path.join(root_dir, output_file), "wb") as f:
        pickle.dump(embeddings, f)

def proc_data_partial(clip_model, split, root_dir="coco"):
    model = ImageEncoder(clip_model, 'cuda')
    model.eval()

    embeddings = {}

    img_dir = os.path.join(root_dir, f"{split}2014")
    i = 0
    for img_name in tqdm(os.listdir(img_dir)):
        if i == 100:
            break
        i += 1
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        img = img.convert("RGB")

        img_num = extract_num(img_name)

        with torch.no_grad():
            img_embedding = model(img)

        img_embedding = img_embedding.cpu().numpy().squeeze()

        embeddings[img_num] = img_embedding

    clip_model = clip_model.replace('/', '_')
    output_file = f"{split}_embeddings_{clip_model}_100.pkl"
    with open(os.path.join(root_dir, output_file), "wb") as f:
        pickle.dump(embeddings, f)

if __name__ == "__main__":

    clip_models = ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"]
    splits = ["train", "val"]

    for clip_model in clip_models:
        for split in splits:
            proc_data(clip_model, split)