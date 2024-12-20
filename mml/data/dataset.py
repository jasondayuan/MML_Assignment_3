"""
    Module contains Dataset class, collate function for DataLoader and loader getter function.

    * ImageCaptionDataset loads data from pickle file and returns image embedding and caption.
    * cl_fn is used to process batch of data and return tensors.
    * get_loader returns DataLoader object.
"""

import os
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, AutoTokenizer
import pickle
import json


class ImageCaptionDataset(Dataset):
    # TODO: 需要实现一个 ImageCaptionDataset
    def __init__(self, clip_model, root_dir):
        self.root_dir = root_dir
        self.split = 'train'

        clip_model = clip_model.replace('/', '_')
        with open(os.path.join(root_dir, f"{self.split}_embeddings_{clip_model}.pkl"), "rb") as f:
            self.data = pickle.load(f)
        
        captions_path = os.path.join(root_dir, 'annotations', 'train_caption.json')
        with open(captions_path, 'r') as f:
            self.captions = json.load(f)

        valid_image_ids = set(self.data.keys())
        self.captions = [item for item in self.captions if int(item['image_id']) in valid_image_ids]
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        img_num = int(self.captions[idx]['image_id'])
        img_emb = self.data[img_num]
        caption = self.captions[idx]['caption']
        img_name = f"COCO_{self.split}2014_{img_num:012}.jpg"
        return img_name, img_emb, caption



def cl_fn(batch, tokenizer):
    # TODO: 需要实现一个 collate function
    _, img_emb, captions = zip(*batch)
    
    img_emb = torch.tensor(np.array(img_emb))
    
    encoding = tokenizer(captions, return_tensors='pt', padding=True)
    
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    return img_emb, input_ids, attention_mask


def get_loader(dataset, bs_exp=5, shuffle=True, num_workers=0, pin_memory=False, text_model="gpt2-medium"):
    if 'gpt2' in text_model:
        tokenizer = GPT2Tokenizer.from_pretrained(text_model)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(text_model)
        tokenizer.pad_token = tokenizer.eos_token

    return DataLoader(
        dataset,
        batch_size=2**bs_exp,
        collate_fn=lambda b: cl_fn(b, tokenizer),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )