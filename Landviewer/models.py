import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn import metrics
from sklearn.cluster import KMeans
import torch.optim as optim
from random import shuffle
import pandas as pd
import numpy as np
import scanpy as sc
from .layers import GraphConvolution



class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Linear(in_size, 1, bias=True)

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class mvGCN(nn.Module):
    def __init__(self, nfeat, nhid, alpha=0.2):
        super(mvGCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nfeat, nhid)
        self.gc3 = GraphConvolution(nfeat, nhid)
        self.attention = Attention(nhid)
        self.gc = GraphConvolution(nfeat, nhid)
        self.nhid = nhid
        #self.mu determined by the init method
        self.alpha = alpha

    def forward(self, x, adj1, adj2, adj3):
        x1 = self.gc1(x, adj1)
        x2 = self.gc2(x, adj2)
        x3 = self.gc3(x, adj3)

        emb = torch.stack([x1, x2, x3], dim=1)
        emb, att = self.attention(emb)
        att = torch.squeeze(att)
        adj = adj1 * att[:, 0] + adj2 * att[:, 1] + adj3 * att[:, 2]

        z = self.gc(emb, adj)
        q = 1.0 / ((1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha) + 1e-8)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return z, emb, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))
        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        #weight = q ** 2 / q.sum(0)
        #return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, X, adj1, adj2, adj3, label=None, lr=0.001, max_epochs=5000, update_interval=3,
            trajectory_interval=50, weight_decay=5e-4, opt="sgd", init="louvain", n_neighbors=10, res=0.4,
            n_clusters=10, random_seed=0, device="cpu"):
        # set_seed(random_seed)
        self.trajectory = []
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "admin":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        X = torch.FloatTensor(X).to(device)
        adj1 = torch.FloatTensor(adj1).to(device)
        adj2 = torch.FloatTensor(adj2).to(device)
        adj3 = torch.FloatTensor(adj3).to(device)
        x1 = self.gc1(X, adj1)
        x2 = self.gc2(X, adj2)
        x3 = self.gc3(X, adj3)
        emb = torch.stack([x1, x2, x3], dim=1)
        emb, att = self.attention(emb)
        att = torch.squeeze(att)
        adj = adj1 * att[:, 0] + adj2 * att[:, 1] + adj3 * att[:, 2]
        features = self.gc(emb, adj)

        #----------------------------------------------------------------        
        if init == "kmeans":
            print("Initializing cluster centers with kmeans, n_clusters known")
            self.n_clusters = n_clusters
            kmeans = KMeans(self.n_clusters, n_init=10, random_state=random_seed)
            #------Kmeans use exp and spatial
            y_pred = kmeans.fit_predict(features.detach().cpu().numpy())

        elif init == "louvain":
            import scanpy as sc
            print("Initializing cluster centers with louvain, resolution = ", res)
            adata = sc.AnnData(features.detach().cpu().numpy())
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            sc.tl.louvain(adata, resolution=res)
            y_pred = adata.obs['louvain'].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
        #----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.nhid)).to(device)
        self.trajectory.append(y_pred)
        features = pd.DataFrame(features.detach().cpu().numpy(), index=np.arange(0, features.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                _, _, q = self.forward(X, adj1, adj2, adj3)
                p = self.target_distribution(q).data
            if epoch % 10 == 0:
                print("Epoch ", epoch)
            optimizer.zero_grad()
            z, emb, q = self(X, adj1, adj2, adj3)
            if label is not None:
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                df = label.to_frame()
                df["ypre"] = y_pred
                df = df.dropna()
                ari = metrics.adjusted_rand_score(df["ypre"], df["Ground Truth"])
                # ari = metrics.adjusted_rand_score(y_pred, label)
                print(epoch, ari)
            # else:
            #     y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            #     sc=metrics.silhouette_score(X.data.cpu().numpy(), y_pred, metric='euclidean')
            #     print(epoch,sc)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            if epoch % trajectory_interval == 0:
                self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

            #Check stop criterion
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
            y_pred_last = y_pred
            # if epoch>0 and (epoch-1)%update_interval == 0 and delta_label < tol:
            #     print('delta_label ', delta_label, '< tol ', tol)
            #     print("Reach tolerance threshold. Stopping training.")
            #     print("Total epoch:", epoch)
            #     break

    def predict(self, X, adj1, adj2, adj3, device):
        z, emb,q = self(torch.FloatTensor(X).to(device),
                                    torch.FloatTensor(adj1).to(device),
                                    torch.FloatTensor(adj2).to(device),
                                    torch.FloatTensor(adj3).to(device))
        return z, emb, q
