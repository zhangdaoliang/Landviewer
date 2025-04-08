#cd Desktop/GCN/pygcn/env2/
import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.sparse import issparse
from anndata import AnnData
import torch
from sklearn.decomposition import PCA
import math
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from . models import *

class stMGC(object):
    def __init__(self):
        super(stMGC, self).__init__()
    def train(self,adata,
            num_pcs=100,
            lr=0.005,
            max_epochs=2000,
            weight_decay=1e-4,
            opt="admin",
            init="louvain", #louvain or kmeans
            n_neighbors=10, #for louvain
            n_clusters=None, #for kmeans
            res=0.4, #for louvain
            random_state=0,
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.num_pcs=num_pcs
        self.res=res
        self.lr=lr
        self.max_epochs=max_epochs
        self.weight_decay=weight_decay
        self.opt=opt
        self.init=init
        self.n_neighbors=n_neighbors
        self.n_clusters=n_clusters
        self.res=res
        self.label=None
        self.adj1 = adata.obsm['adjl']
        self. adj2 = adata.obsm['adji']
        self.adj3 = adata.obsm['adjx']
        self.device=device
        assert adata.shape[0] == self.adj1.shape[0]== self.adj1.shape[1]
        assert adata.shape[0] == self.adj2.shape[0] == self.adj2.shape[1]
        assert adata.shape[0] == self.adj3.shape[0] == self.adj3.shape[1]
        pca = PCA(n_components=self.num_pcs,random_state=random_state)
        if issparse(adata.X):
            pca.fit(adata.X.A)
            embed=pca.transform(adata.X.A)
        else:
            pca.fit(adata.X)
            embed=pca.transform(adata.X)

        set_seed(random_state)
        if "Ground Truth" in adata.obs.columns:
            self.label=adata.obs["Ground Truth"]
        ###------------------------------------------###
        #----------Train model----------
        self.model=mvGCN(embed.shape[1],embed.shape[1]).to(device)
        self.model.fit(embed,self.adj1,self.adj2,self.adj3,label=self.label,lr=self.lr,max_epochs=self.max_epochs,
                       weight_decay=self.weight_decay,opt=self.opt,init=self.init,random_seed=random_state,
                       n_neighbors=self.n_neighbors,n_clusters=self.n_clusters,res=self.res, device=device)
        self.embed=embed


    def predict(self):
        z,emb, q=self.model.predict(self.embed,self.adj1,
                               self.adj2,self.adj3,self.device)
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        # Max probability plot
        # prob=q.detach().cpu().numpy()
        return y_pred, z.detach().cpu().numpy()



