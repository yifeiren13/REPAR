#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import pickle
import datetime
import argparse
from pathlib import Path
#import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from tqdm import tqdm
try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter

from models import RebarPARAFAC2,SmoothnessConstraint, TemporalDependency
from utils import EarlyStopping, AverageMeter, PaddedDenseTensor
import numpy as np
import pandas as pd
#import time
#from matplotlib import pyplot as plt
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

 
#trainmimic = pickle.load( open ("myDicts.p", "rb") )
trainmimic = pickle.load( open ("extract5000vital.p", "rb") )
#trainmimic2 = pickle.load( open ("169525mis30.p", "rb") )
'''
def logistic_regression(Wtrain, y_train, Wtest, y_test):
    classifier = LogisticRegressionCV(cv=5, Cs=10, class_weight='balanced')
    classifier.fit(Wtrain, y_train)
    pred_prob = classifier.predict_proba(Wtest)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_prob[:, 1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc
'''


def logistic_regression(Wtrain, y_train, Wtest, y_test):
    classifier = LogisticRegressionCV(cv=5, Cs=10, class_weight='balanced')
    classifier.fit(Wtrain, y_train)
    pred_prob = classifier.predict_proba(Wtest)
    prauc = average_precision_score(y_test, pred_prob[:,1])
    #fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_prob[:, 1], pos_label=1)
    #auc = metrics.auc(fpr, tpr)
    return prauc

def predictionauc(W,label):
    best_auc = []
    for i in range(5):
        Wtrain, Wtest, labels_train, labels_test = train_test_split(W, label, 
                                                                    train_size=0.8, test_size=0.2)
        best_auc.append(logistic_regression(Wtrain, labels_train, Wtest, labels_test))
        
    return max(best_auc)



def validatesquarederror(model, dataloader):
    with torch.no_grad():
        targets = []
        predictions = []
        for pids, Xdense, masks in dataloader:
            pids, Xdense, masks = pids.to(device), Xdense.to(device), masks.to(device)
            output = model(pids)
            #output = output[masks==1]
            #target = Xdense[masks==1]
            output = output
            target = Xdense
            targets.append(target)
            predictions.append(output)
        targets = torch.cat(targets, dim=0)
        predictions = torch.cat(predictions, dim=0)
        normX = (targets**2).sum()
        me = ((targets-predictions)**2).sum()

        fit =1-(me/normX)
    return fit

def oldscore(data,ydata):
    X_train, X_test, y_train, y_test = train_test_split(data, ydata, test_size=0.3, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    #y_pred = logreg.predict(X_test)
    #print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
    return logreg.score(X_test, y_test)

def singulart(matrix,threshS):
    #threshS = betaL(d) / rhoL
    #threshS = 0.01
    U,S,V = torch.svd(matrix)  
    nnzsv = len(S[S>threshS])
    if  nnzsv >= 1:
        SS = S[0:nnzsv] - threshS;
        smat = torch.zeros_like(S)
        smat[0:nnzsv] = SS
        result = torch.nn.Parameter(torch.mm(torch.mm(U, torch.diag(smat)), torch.transpose(V, 0, 1)))
    else:
        result = torch.nn.Parameter(torch.zeros_like(matrix))
    return result

def singularte(matrix,threshS):
    #threshS = 0.01
    uu,ss,vv = torch.svd(matrix)
    qq = ss-threshS
    qq[qq < 0] = 0
    smat = torch.diag_embed(qq)
    vw = torch.transpose(vv, 1, 2)
    result = torch.nn.Parameter(torch.matmul(torch.matmul(uu, smat), vw))
    return result
    
    

    
    
def train_rebar_parafac2(indata, 
                            num_visits, 
                            num_feats, 
                            log_path, 
                            pos_prior, 
                            reg_weight, 
                            smooth_weight,
                            rank, 
                            weight_decay, 
                            alpha, 
                            gamma, 
                            lr, 
                            seed, 
                            batch_size, 
                            smooth_shape,
                            iters, 
                            patience, 
                            label):

    if seed is not None:
        torch.manual_seed(seed)
    '''
    UU = nn.Parameter(torch.zeros(3000, max(num_visits), rank))
    dfU= pd.read_csv(r"/Users/yifeiren/Desktop/MIMICExtract/U3000.csv")
    dfV = pd.read_csv(r"/Users/yifeiren/Desktop/MIMICExtract/V3000.csv")
    dfW = pd.read_csv(r"/Users/yifeiren/Users/yifeiren/Desktop/MIMICExtract-REBAR/mainmimic.py/Desktop/MIMICExtract/W3000.csv")
    dfheight = pd.read_csv(r"/Users/yifeiren/Desktop/MIMICExtract/height3000.csv")
    heightvalue = dfheight.values
    Uvalue = dfU.values
    ro = 0
    for i in range(3000):
        height = int(heightvalue[i])
        UU[i][0:height,:] = torch.from_numpy(Uvalue[ro:ro+height,:])
        ro = ro + height
    VV = nn.Parameter(torch.zeros(104, 20))
    Vvalue = dfV.values   
    VV =  torch.from_numpy(Vvalue)
    WW = nn.Parameter(torch.zeros(3000, 20))
    Wvalue = dfW.values   
    WW=  torch.from_numpy(Wvalue)
'''
    model = RebarPARAFAC2(num_visits, 
                             num_feats, 
                             rank, 
                             alpha=alpha, 
                             gamma=gamma).to(device)
    #model.S = torch.nn.Parameter(WW)
    #model.U = torch.nn.Parameter(UU)
    #model.V = nn.parameter(torch.from_numpy(Vvalue))
    #smoothness = SmoothnessConstraint(beta=smooth_shape)
    #temp_model = self.temporal_type(self.rank, nlayers, nhidden, dropout)
    temp_model = TemporalDependency(rank=50,nlayers=1, nhidden=100, dropout=0)
    #tf_loss_func = PULoss(prior=pos_prior)
    tf_loss_func = nn.MSELoss(reduction = 'sum')
    #tf_loss_func = nn.PoissonNLLLoss()
    #tf_loss_func = PoiLoss()

    optimizer_pt_reps = torch.optim.SGD([model.U, model.S], 
                             lr=lr, momentum=0.9, dampening=0, weight_decay=weight_decay, nesterov=True)
    optimizer_phenotypes = torch.optim.SGD([model.V], lr=lr, momentum=0.9, dampening=0, weight_decay=weight_decay, nesterov= True)
    
    optimizer_temp = torch.optim.SGD(temp_model.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=weight_decay,nesterov= True)

    lr_scheduler_pt_reps = ReduceLROnPlateau(optimizer_pt_reps, 
                                             mode='max', 
                                             cooldown=10,
                                             patience = 20,
                                             min_lr=1e-6)
    lr_scheduler_phenotypes = ReduceLROnPlateau(optimizer_phenotypes, 
                                                mode='max', 
                                                cooldown=10, 
                                                patience =20,
                                                min_lr=1e-6)
    
    #lr_scheduler_temp = ReduceLROnPlateau(optimizer_temp, 
    #                                            mode='max', 
    #                                            cooldown=10, 
    #                                            patience = 20,
    #                                            min_lr=1e-6)

    #writer = SummaryWriter(log_path)

    collators = [PaddedDenseTensor(indata, num_feats, subset=subset) 
                 for subset in ('train', 'validation', 'test')]
    loaders = [DataLoader(TensorDataset(torch.arange(len(num_visits))), 
                          shuffle=True, 
                          num_workers=2, 
                          batch_size=batch_size, 
                          collate_fn=collator)
               for collator in collators]
    train_loader, valid_loader, test_loader = loaders

    #early_stopping = EarlyStopping(patience=patience)


    for epoch in range(iters):

        epoch_uni_reg = AverageMeter()

        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}')
        lr = optimizer_pt_reps.param_groups[0]['lr']
        #optimizer_temp.zero_grad()
        if epoch < 10:
            thredS = 0.01
            thredU = 0.01
            thredV = 0.01
        else:
            thredS = 0.001
            thredU = 0.001
            thredV = 0.001
        #aucscore = 0
        for pids, Xdense, masks  in train_loader:
            num_visits_batch = masks.squeeze(-1).sum(dim=1).to(device)

            num_visits_batch, pt_idx = num_visits_batch.sort(descending=True)
            pids = pids[pt_idx].to(device)
            Xdense = Xdense[pt_idx].to(device)
            masks = masks[pt_idx].to(device)

            #deltas = deltas[pt_idx].to(device) / 7  # transform days to weeks

            # update U & S
            model.S.requires_grad = True
            model.U.requires_grad = True
            model.V.requires_grad = False
            
            optimizer_pt_reps.zero_grad()
            
            output = model(pids)
            #loss, out = tf_loss_func(output, Xdense, masks=masks)
            out = tf_loss_func(output, Xdense)
            print(f'tensorloss: {out:.3f}')
            uni_reg = model.uniqueness_regularization(pids)
            out = out + reg_weight * uni_reg

            smoothness_reg = temp_model(model.U[pids])
                                        #num_visits_batch,
                                        #deltas=deltas)

            print(f'smoothloss: {smoothness_reg:.3f}')
            out = out + smooth_weight * smoothness_reg
            #print(f'addedauc: {aucscore:.3f}')
            out.backward()
            optimizer_pt_reps.step()
            model.projection()
            
            
            
            epoch_uni_reg.update(uni_reg.item(), n=pids.shape[0])

            # update V
            model.S.requires_grad = False
            model.U.requires_grad = False
            model.V.requires_grad = True
            optimizer_phenotypes.zero_grad()
            output = model(pids)
            #loss, out = tf_loss_func(output, Xdense, masks=masks)
            out = tf_loss_func(output, Xdense)
            out.backward()
            optimizer_phenotypes.step()
            model.projection()
            
            
            

            #epoch_tf_loss.update(loss.item(), n=masks.sum())

            #epoch_tf_loss.update(out.item(), n=masks.sum())
            pbar.update()
            
            
            
            # update temporal model
            optimizer_temp.zero_grad()
            temporal_loss = temp_model(model.U[pids])
            temporal_loss.backward()
            optimizer_temp.step()


        model.update_phi()
        for i, num_visit in enumerate(num_visits):
            model.U.data[i, num_visit:] = 0
            
        Sbefore = model.S.sum()
        print(f'Sbefore: {Sbefore:.3f}')
        model.S = singulart(model.S,thredS)
        Safter = model.S.sum()
        print(f'Safter: {Safter:.3f}')
        Ubefore = model.U.sum()
        print(f'Ubefore: {Ubefore:.3f}')
        model.U = singularte(model.U,thredU)
        Uafter = model.U.sum()
        print(f'Uafter: {Uafter:.3f}')
        
        Vbefore = model.V.sum()
        print(f'Vbefore: {Vbefore:.3f}')
        model.V = singulart(model.V,thredV)
        Vafter = model.V.sum()
        print(f'Vafter: {Vafter:.3f}')
        
        W = model.S.detach().numpy() 
        #aucscore = predictionauc(W,label)
        aucscore = predictionauc(W,label)
        
        #ap_valid = validate(model, valid_loader)
        print(f'PR-AUC: {aucscore:.3f}')
        
        fit = validatesquarederror(model, valid_loader)
        print(f'FIT: {fit:.5f}')
        print(f'lr: {lr:.5f}')
        lr_scheduler_pt_reps.step(fit)
        lr_scheduler_phenotypes.step(fit)


        #if fit > fitscoreee:
        #    fitscoreee = fit
        #    fitcount = 0
        #else:
        #    fitcount = fitcount + 1
        #    if fitcount > 15:
        #        print('Early Stopped.')
        #        break

                

    return model.U, model.S, model.V,model,valid_loader




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str,
                        help='Name of the experiment.')
    parser.add_argument('--data_path', '-d', type=str, 
                        default='./demo_data.pkl',
                        help='The path of input data.')
    parser.add_argument('--pi', '-p', type=float, default=0.005,
                        help='Class prior for the positive observations.')
    parser.add_argument('--uniqueness', '-u', type=float, default=1e-3,
                        help='Weighting for the uniqueness regularization.')
    parser.add_argument('--rank', '-r', type=int, default=50,
                        help='Target rank of the PARAFAC2 factorization.')
    parser.add_argument('--seed', type=int, 
                        help='Random seed')
    parser.add_argument('--alpha', type=float, default=4,
                        help='Maximam infinity norm allowed for the factor '\
                             'matrices.')
    parser.add_argument('--gamma', type=float, default=1,
                        help='Shape parameter for the sigmoid function.')
    parser.add_argument('--lr', type=float, default=1e-1,
                        help='Learning rate of the optimizers.')
    parser.add_argument('--wd', type=float, default=0,
                        help='Weight decay for the optimizers.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default = 100)
    parser.add_argument('--proj_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20,
                        help='Epochs to wait before early stopping, use 0 to '\
                             'switch off early stopping.')
    parser.add_argument('--smooth', type=float, default=1,
                        help='Weighting for the smoothness regularization.')
    parser.add_argument('--smooth_shape', type=float, default=1,
                        help='shape parameter for the time-aware TV smoothing')
    parser.add_argument('--out', type=str, default='./results/',
                        help='Directory to save the results, a subfolder will '\
                             'be created.')
    parser.add_argument('--log', type=str, default='./results/tb_logs',
                        help='Path to store the tensorboard logging file.')

    args = parser.parse_args()

        
    ##########################
    ## Load data #############
    ##########################
    indata = trainmimic
    data_train = indata
    #train_idx, test_idx = train_test_split(range(len(indata)), 
    #                                              train_size=0.8)
    #xx = 0
    #for pt in range(len(indata)):
        #print(pt)
        #test = indata[pt]
        #yy = indata[pt]['train'][:,1].max()
        #if yy > xx:
            #xx = yy

    num_feats = max([pt['train'][:, 1].max() + 1 for pt in indata])
    num_feats= int(num_feats)
    #num_feats = 14
        
    #data_train = [indata[x] for x in train_idx]
    num_visits_train = [pt['times'] for pt in data_train]

    #data_test = [indata[x] for x in test_idx]
    #num_visits_test = [pt['times'] for pt in data_test]
    


    ##################################
    ## Set up experiment #############
    ##################################
    exp_id = f'REPAR_rank{args.rank}'
    if args.name is not None:
        exp_id = args.name + '_' + exp_id
    if args.seed is not None:
        exp_id += f'_seed{args.seed}'
    exp_id += f'_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
    results_out_dir = Path(args.out) / exp_id

    results_out_dir.mkdir(parents=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    with open(results_out_dir/'config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    ###############################
    ## Run experiment #############
    ###############################
    print(f'rank={args.rank}, seed={args.seed}')
    labeldata = []
    for i in range(5000):
        labeldata.append(data_train[i]["label"])
 
    U,S,V,model,valid_loader = train_rebar_parafac2(
        data_train, num_visits_train, num_feats, 
        patience=args.patience or args.epochs, 
        log_path=os.path.join(args.log, exp_id),
        pos_prior=args.pi,
        smooth_weight=args.smooth,
        seed=args.seed,
        rank=args.rank,
        reg_weight=args.uniqueness,
        weight_decay=args.wd,
        alpha=args.alpha,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        smooth_shape=args.smooth_shape,
        iters=args.epochs,
        label=labeldata)


