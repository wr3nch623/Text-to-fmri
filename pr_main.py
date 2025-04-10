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
from math import log

import model
from model import SqueezeNet
from torch.optim.lr_scheduler import StepLR, ExponentialLR

import torch.optim as optim
from transformers import CLIPProcessor, CLIPModel
import transformers

from torchviz import make_dot
import hiddenlayer as hl

class BasicBlock(nn.Module):
    def __init__(self, levelDIM, outputDIM):
        super(BasicBlock, self).__init__()

        self.lin1 = nn.Linear(levelDIM, levelDIM)
        self.relu = nn.LeakyReLU()

        self.bn1 = nn.BatchNorm1d(levelDIM)
        self.lin2 = nn.Linear(levelDIM, levelDIM)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(levelDIM)

        self.weight = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        initial = x
        x = self.lin1(x)
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.bn2(x)
        x += initial
        x = self.relu2(x)

        return x


class DumbNNModel(nn.Module):
    def __init__(self, hiddenDIM, levelDIM, outputDIM):
        super(DumbNNModel, self).__init__()
        
        layers = []

        layers.append(nn.Linear(32, 64))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(64, levelDIM))
        layers.append(nn.LeakyReLU())


        for i in range(hiddenDIM):
            layers.append(BasicBlock(levelDIM, levelDIM))
            #if i % 5 == 0 and i > 4:
            #    layers.append(nn.BatchNorm1d(128))

        layers.append(nn.Linear(levelDIM, outputDIM))

        self.output_head_1 = nn.Sequential(*layers)
        




    def forward(self, x):
        #print(x.dtype)
        x=x.float()
        x.type(torch.DoubleTensor)
        #print(x.shape)
        
        #out1 = self.shared_layers(x)  # Pass through shared layers
        #print(x.dtype)
        
        out1 = self.output_head_1(x)  # First output
        #out2 = self.output_head_2(x)  # Second output
        #print(out1.dtype)
        #print(out2.dtype)
        #return out1, out2  # Return both outputs
        return out1


def getArrays(ds):
    return ds['id'].to_numpy(), ds['caption'].to_numpy()

def get_order_fmri(idx, fmri):
    r = []
    for i in range(len(idx)):
        r.append(fmri[idx[i]])

    return r

def get_tokenized_data(tokenizer, captions, lh_fmri):
    tokenized_captions = []
    first_lh_fmri = []
    for i in range(len(captions)):
        tokenized_captions.append(tokenizer(captions[i], return_tensors='pt', padding='max_length', truncation=False, max_length=32)['input_ids'])
        first_lh_fmri.append(torch.tensor(lh_fmri[i]))

    return tokenized_captions, first_lh_fmri

def debug(cap, fmri):
    cp = cap.detach().cpu().numpy()

def generate_loss_graph(loss, filename):
    logloss = []
    for x in loss:
        logloss.append(log(x))

    fig, axs = plt.subplots(2, 1, figsize=(10, 5))

    axs[0].plot(loss[100:], label='Loss per Batch')
    axs[0].set_xlabel('Batch Number')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss per Batch during Training')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(logloss[100:], label='Logarithmic loss per Batch')
    axs[1].set_xlabel('Batch Number')
    axs[1].set_ylabel('Logarithmic loss')
    axs[1].set_title('Logarithmic loss per Batch during Training')
    axs[1].legend()
    axs[1].grid()


    plt.tight_layout()

    plt.savefig(filename)

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

    # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
        'mapping_floc-faces.npy', 'mapping_floc-places.npy',
        'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(MASK, r),
            allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
        'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
        'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
        'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
        'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
        'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
        'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(MASK,
            lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(MASK,
            rh_challenge_roi_files[r])))


    # Select the correlation results vertices of each ROI
    roi_names = []
    lh_roi_masks = []
    rh_roi_masks = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                lh_roi_masks.append(lh_roi_idx)
                rh_roi_masks.append(rh_roi_idx)
    
    print(f'len first roi: {len(lh_roi_masks[0])}')

    #debug
    gradPrint = False
    debug = True
    allloss = []
    print(lh_fmri)

    #early stopping
    patience = 700
    best_loss = 9999999
    epochs_without_improvement = 0


    ds = getDataset(Path('..') /'algonauts23'/ 'dataset' / 'algonauts_2023_challenge_data','training', 'subj01')
    print(ds)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DumbNNModel(1, 64,len(lh_roi_masks[0]))

    loss_fn1 = nn.L1Loss()
    loss_fn2 = nn.L1Loss()
    
    
    
    #optimizer = optim.AdamW(model.parameters(), lr=0.5, maximize = False)
    #optimizer = optim.AdamW(model.parameters(), lr=0.05, maximize = False)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, maximize = False)

    config = transformers.CLIPTextConfig()
    tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    

    model.to(device)
    
    batch_size = 256
    
    #loading training data
    idx, captions = getArrays(ds)
    print(idx)
    tokCap, tok_lh_fmri = get_tokenized_data(tokenizer, captions, get_order_fmri(idx, lh_fmri[:, lh_roi_masks[0]]))
    trainingData = TrainingData(tokCap, tok_lh_fmri)

    dataLoader = DataLoader(trainingData, batch_size = batch_size, shuffle = False)

    # Training loop 50 epochs
    num_epochs = 200
    
    
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5) 
    #scheduler = ExponentialLR(optimizer, gamma=0.5) 

    print('training start')
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        first = False
        for cap, lh_fmri_c in dataLoader:
            cap = cap.to(device)
            #print(cap)
            lh_fmri_c = lh_fmri_c.to(device)
            cap = cap.squeeze(1)
            

            output1 = model(cap)

            print(output1.shape)
            print(lh_fmri_c.shape)
            
            optimizer.zero_grad()
            loss1 = loss_fn1(output1, lh_fmri_c)
            #ot = output1.cpu().detach().numpy()[0][0][0]
            #ac = lh_fmri_c.cpu().detach().numpy()[0]
            loss = loss1
            loss.backward()
            optimizer.step()
            
            allloss.append(loss.item())
            print(f'output: {output1}')
            print(lh_fmri_c)
            print()
            #break

            #early stopping: need to implement validation set to use it correctly, for now it will be only on training set
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), 'best_checkpoint.pt')
                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1

            #print(f'output : {output1}')
            


            #if debug:
            #    for i in range(len(lh_fmri_c)):
            #        #ot = output1.cpu().detach().numpy()[i][0][0]
            #        #ot = float(output1.cpu().detach().numpy()[i])
            #        ot = output1.cpu().detach().numpy()[i]
            #        ac = lh_fmri_c.cpu().detach().numpy()[i]
            #        temp = output1.cpu().detach()
            #        #print(output1)
            #        #print(lh_fmri_c)
            #        #if(abs(ot - ac) >= 0.5):

            #        #    print(f'(*) output: {ot:4}, actual: {ac:4}')
            #        #else:
            #        print(f'output: {ot}, actual: {ac}')

            print(f'loss: {loss:4f}')

            print()
            #print(f'output: {ot:4}, actual: {ac:4}, loss: {loss:4}')
            #print(f'loss: {loss1:4}, output: {out}')
            #print(f'loss: {loss:4}')
            if first or debug:
                first = False
                #print(f'output: {ot:4}, actual: {ac:4}, loss: {loss:4}')
                if gradPrint == True:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            print(f'Gradient for {name}: {param.grad}')

        if epochs_without_improvement >= patience:
            print('Early stopping used')
            print(f'Final epoch : {epoch}')
            break

        #scheduler.step()
        print()
        print(epoch)
        print()

    print("Training complete!")

    #print(allloss)

    generate_loss_graph(allloss, 'training_loss.png')

    #model = DumbNNModel(32, len())
    model.load_state_dict(torch.load('best_checkpoint.pt'))
    model.to(device)
    model.eval()

    ds = sortDataframe(ds)

    lh_pred = []
    rh_pred = []

    val_loss = []
    for cap, lh_fmri_c in dataLoader:
        cap = cap.to(device)
        lh_fmri_c = lh_fmri_c.to(device)
        cap = cap.squeeze(1)

        #lh_fmri_c = lh_fmri_c.unsqueeze(1)
        

        output1 = model(cap)
        
        print(lh_fmri_c.shape)
        print(output1.shape)

        optimizer.zero_grad()
        loss1 = loss_fn1(output1, lh_fmri_c)
        #ot = output1.cpu().detach().numpy()[0][0][0]
        #ac = lh_fmri_c.cpu().detach().numpy()[0]
        loss = loss1
        
        val_loss.append(loss.item())



        if debug:
            for i in range(len(lh_fmri_c)):
                #ot = output1.cpu().detach().numpy()[i][0][0]
                #ot = float(output1.cpu().detach().numpy()[i])
                ot = output1.cpu().detach().numpy()[i][0]
                ac = lh_fmri_c.cpu().detach().numpy()[i][0]
                temp = output1.cpu().detach()
                #print(output1)
                #print(lh_fmri_c)
                #if(abs(ot - ac) >= 0.5):

                #    print(f'(*) output: {ot:4}, actual: {ac:4}')
                #else:
                #    print(f'output: {ot:4}, actual: {ac:4}')

            print(f'loss: {loss:4f}')
    print(val_loss)
    generate_loss_graph(val_loss, 'validation_loss.png')


if __name__ == '__main__':
    main()
