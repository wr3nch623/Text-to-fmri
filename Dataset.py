import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr
import json
import statsmodels.api as sm
import pandas as pd


# Load data
# load everything useful
# logging

json_loc = '/media/data1/nsd_augmentation_data/metadata.json'
path_prefix = "/media/data1/algonauts_2023_challenge_data"

DICT = ['person', 'man']

def getDataset(algonauts_dir, parent_split, sub):
    with open(json_loc) as json_data:
        raw_data = json.load(json_data)

    images = []
    image_path = []
    image_number = []
    df = pd.DataFrame(columns=['id', 'caption'])
    if (parent_split == 'training'):
        split = 'train'
    else:
        split = 'val'
    count = 0
    for data in raw_data['images']:
        check_caption = False
        path = data['path'].removeprefix(path_prefix)

        p = path.split('/')
        #print(p)
        if any(parent_split in dono for dono in p):
            if p[1] == sub:
                for word in DICT:
                    if any(word in phrase for phrase in data['captions']):
                        check_caption = True
                        ptemp = Path(path) 
                        entities = dict(
                            [part.split("-", maxsplit=2) for part in ptemp.stem.split("_")])
                      
                        #print([int(list(entities.values())[0])-1])
                        #print([data['captions'][1]])
                        pdtemp = pd.DataFrame([[int(list(entities.values())[0]) - 1, data['captions'][0]]], columns=['id', 'caption'])

                        #df.concat(pdtemp, ignore_index=True)
                        df = pd.concat([df, pdtemp], ignore_index=True)
                        count += 1
                            
        
        #print(check_caption)
        if check_caption:
            images.append(data['path'])
            #print(data['path'])
            #print('added')


        #if count >= 10:
        #    break

    #print("images")
    #print(images)
    image_dir = algonauts_dir / sub / f"{parent_split}_split" / f"{parent_split}_images"
    #print(image_dir)
    for i in images:
        image_path.append(i)

    
    #print("image path HERE!!!")
    #print(image_path)

    image_list = sorted(image_dir.glob("*.png")) 

    #print("IMAGE PRINT HERE")
    #for i in image_path:
        #print(i)

    #return sorted(image_path)
    return df

def sortDataframe(df):
   return df.sort_values('id')


class TrainingData(Dataset):
    def __init__ (self, cap, lh_fmri):
        self.cap = cap
        self.lh_fmri = lh_fmri

    def __len__(self):
        return len(self.cap)

    def __getitem__(self, idx):
        return self.cap[idx], self.lh_fmri[idx]

    def getitem(self, idx):
        return self.cap[idx], self.lh_fmri[idx]




if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    #indices = getImageList(Path('.') / 'algonauts_2023_challenge_data','training', 'subj01') 
    df = getDataset(Path('..') /'algonauts23'/ 'dataset' / 'algonauts_2023_challenge_data','training', 'subj01')
    print(df)
    #df = sortDataframe(df)
    #print(df)
    print('list')
    ids = df['id'].to_numpy()
    cap = df['caption'].to_numpy()
    
    print(ids)
    print(cap)
    #dataloader = DataLoader(df, batch_size=10)

    #for batchdata, batchlabels in dataloader:
    #    print(batchdata)
    #    print(batchlabels)
    #    print()
