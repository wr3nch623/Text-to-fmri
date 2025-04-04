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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr
import json
import statsmodels.api as sm
from Dataset import *

from torch.optim.lr_scheduler import StepLR

import torch.optim as optim
from transformers import CLIPProcessor, CLIPModel
import transformers

from torchviz import make_dot
import hiddenlayer as hl

class DumbNNModel(nn.Module):
    #def __init__(self, input_dim, output1_dim, output2_dim, nhead, num_encoding_layers, num_decoding_layers, dim_feedforward, dropout=0.1):
    def __init__(self):
        super(DumbNNModel, self).__init__()
        
        self.transformer = nn.Transformer(d_model = 1024, nhead = 8, num_encoder_layers = 6, num_decoder_layers = 6, batch_first = True)
        
        self.input_layer = nn.Linear(512,1024)

        # Output Head 1 (19004-dimensional)
        self.output_head_1 = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.InstanceNorm1d(4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 19004)  # Final output size
        )
        
        # Output Head 2 (20554-dimensional)
        self.output_head_2 = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.InstanceNorm1d(4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 20544)  # Final output size
        )

    def forward(self, x):
        print(x.shape)
        #x = x.unsqueeze(0)
        x = self.input_layer(x)
        x = x.unsqueeze(0)

                # Pass through the transformer
        transformer_output = self.transformer(x, target)

        # Take the output from the last time step
        x = transformer_output[-1, :, :]  # Get the last output (shape: [batch_size, feature_size])

        # Pass through the output layers
        output1 = self.output_layer1(x)
        output2 = self.output_layer2(x)

        return output1, output2



def main():
    LOC = Path('../algonauts23/dataset')

    SUBJ = 'subj01'
    PATH = LOC / 'algonauts_2023_challenge_data' / SUBJ / 'training_split' / 'training_fmri'
    MASK = LOC / 'algonauts_2023_challenge_data' / SUBJ / 'roi_masks'

    lh_path = PATH / "lh_training_fmri.npy"
    rh_path = PATH / 'rh_training_fmri.npy'
    lh_fmri = np.load(lh_path)
    rh_fmri = np.load(rh_path)

    lh_mask_path = MASK / 'lh.all-vertices_fsaverage_space.npy'
    rh_mask_path = MASK / 'rh.all-vertices_fsaverage_space.npy'
    lh_mask = np.load(lh_mask_path)
    rh_mask = np.load(rh_mask_path)

    print(lh_fmri)

    ds = getDataset(Path('..') /'algonauts23'/ 'dataset' / 'algonauts_2023_challenge_data','training', 'subj01')
    print(ds)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DumbNNModel()

    loss_fn1 = nn.MSELoss()
    loss_fn2 = nn.MSELoss()
    
    print(model.parameters())
    optimizer = optim.AdamW(model.parameters(), lr=0.0000001)

    config = transformers.CLIPTextConfig()
    #print(config)
    tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


    model.to(device)

    # Training loop
    num_epochs = 2
    
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1) 

    print('training start')
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        
        #for inputs, targets1, targets2 in dataloader:
        for i in range(len(ds)):
            
            #print(el)
            #break
            #inputs, targets1, targets2 = inputs.to(device), targets1.to(device), targets2.to(device)
            
            inputs = tokenizer(ds.iloc[i]['caption'], return_tensors='pt', padding='max_length', truncation=True,max_length=512)['input_ids'].to(device)
            #print(ds.iloc[i]['caption'])
            #print(inputs)
            #print(type(inputs))
            #target1 = torch.from_numpy(lh_fmri[ds.iloc[i]['id']]).to(device)
            #target2 = torch.from_numpy(rh_fmri[ds.iloc[i]['id']]).to(device)
            target1 = torch.Tensor(lh_fmri[ds.iloc[i]['id']]).type(torch.float64).to(device)
            target2 = torch.Tensor(rh_fmri[ds.iloc[i]['id']]).type(torch.float64).to(device)
            

            target1 = target1.unsqueeze(0)
            target2 = target2.unsqueeze(0)
            #target1 = lh_fmri[ds.iloc[i]['id']]
            #target2 = rh_fmri[ds.iloc[i]['id']]
            
            #print(target1)
            #print(target2)

            #print(len(output1))
            #print(len(output2))
            #break

            optimizer.zero_grad()  # Reset gradients
            #print(inputs)
            output1, output2 = model(inputs)  # Forward pass

            #print(len(target1))
            #print(len(output1))
            #print(output1)

            #print(target2.shape)
            #print(output2.shape)
            #print(output2)


            loss1 = loss_fn1(output1, target1)  # Compute loss for first output
            loss2 = loss_fn2(output2, target2)  # Compute loss for second output

            loss = loss1 + loss2  # Combine losses
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model weights
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Loss1: {loss1:.4}, Loss2: {loss2:.4}")
        
        scheduler.step()

    print("Training complete!")

    model.eval()
    ds = sortDataframe(ds)

    lh_pred = []
    rh_pred = []
    for i in range(len(ds)):


        inputs = tokenizer(ds.iloc[i]['caption'], return_tensors='pt', padding='max_length', truncation=True,max_length=512)['input_ids'].to(device)
        target1 = torch.Tensor(lh_fmri[ds.iloc[i]['id']]).to(device)
        target2 = torch.Tensor(rh_fmri[ds.iloc[i]['id']]).to(device)
        

        target1 = target1.unsqueeze(0)
        target2 = target2.unsqueeze(0)
        optimizer.zero_grad()  # Reset gradients
        output1, output2 = model(inputs)  # Forward pass

        #lh_pred.append(output1.to(torch.device('cpu')))
        #rh_pred.append(output2.to(torch.device('cpu')))

        lh_pred.append(output1.cpu().detach().numpy()[0].tolist())
        #print(output1.cpu().detach().numpy()[0])
        rh_pred.append(output2.cpu().detach().numpy()[0].tolist())



        loss1 = loss_fn1(output1, target1)  # Compute loss for first output
        loss2 = loss_fn2(output2, target2)  # Compute loss for second output

        loss = loss1 + loss2  # Combine losses

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Loss1: {loss1:.4}, Loss2: {loss2:.4}")

    print(lh_pred)

    np.save('lh_pred.npy', lh_pred)
    np.save('rh_pred.npy', rh_pred)

    #inputs = tokenizer(ds.iloc[2]['caption'], return_tensors='pt', padding='max_length', truncation=True,max_length=512)['input_ids']


    #print(ds.iloc[2])
    #print(inputs)
    #print(inputs.shape)



if __name__ == '__main__':
    main()
