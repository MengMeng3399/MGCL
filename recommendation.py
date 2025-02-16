#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.argv = ['run.py']

import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum
from torch.nn.parameter import Parameter
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from time import time
from prettytable import PrettyTable
import logging
from utils.parser import parse_args
import dgl
import dgl.data
from utils.data_loader import *
from model import *

import logging


# In[2]:


the_k=10
global args
args = parse_args()
device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")


# In[3]:


seed = 1002356
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[4]:


args.dim*2


# In[5]:


from dgl.data.utils import load_graphs
glist,_ = load_graphs("./data/final_data/HGA/graph.bin") 
# glist, label_dict = load_graphs("./data/graph.bin", [0]) # glist will be [g1]
bi_glist,_ = load_graphs("./data/final_data/HGA/bi_graph.bin") 


# In[6]:


train_g,train_pos_g,train_neg_g,warm_test_pos_g,warm_test_neg_g,cold_test_pos_g,cold_test_neg_g,all_test_pos_g,all_test_neg_g,tag_graph,senmantic_graph = glist[:11]

bi_train_g,bi_train_pos_g,bi_train_neg_g,bi_warm_test_pos_g,bi_warm_test_neg_g,bi_cold_test_pos_g,bi_cold_test_neg_g,bi_all_test_pos_g,bi_all_test_neg_g,bi_tag_graph,bi_senmantic_graph = bi_glist[:11]


# In[7]:


num_mashup=int(senmantic_graph.num_edges()/the_k)
num_nodes=int(train_g.num_nodes())
num_api=num_nodes-num_mashup

n_params={
"n_mashup":num_mashup,
"n_nodes":num_nodes,
"n_api":num_api
}


# In[8]:


def to_bidirectional(g):
    src,dst=g.edges()
    new_src=torch.cat([src,dst])
    new_dst=torch.cat([dst,src])
    bidirectional_g = dgl.graph((new_src, new_dst), num_nodes=g.num_nodes())
    return bidirectional_g

def reverse_directional(g):
    src,dst=g.edges()
    reverse_g = dgl.graph((dst, src), num_nodes=g.num_nodes())
    return reverse_g


# In[9]:


# 为计算NDCG,HR
# cold:
u,v=cold_test_pos_g.edges()
unique_u = torch.unique(u)
src = unique_u.repeat(num_api)

num_repeat=len(unique_u)
print(num_repeat)
dst = torch.tensor([0+num_mashup]*num_repeat)

for i in range(1,num_api):
    single_dst=torch.tensor([i+num_mashup]*num_repeat)
    dst=torch.cat([dst,single_dst])

all_cold_graph=dgl.graph((src,dst))

# warm:

u,v=warm_test_pos_g.edges()

unique_u = torch.unique(u)
src = unique_u.repeat(num_api)

num_repeat=len(unique_u)
print(num_repeat)
dst = torch.tensor([0+num_mashup]*num_repeat)

for i in range(1,num_api):
    single_dst=torch.tensor([i+num_mashup]*num_repeat)
    dst=torch.cat([dst,single_dst])

all_warm_graph=dgl.graph((src,dst))

# all:

u,v=all_test_pos_g.edges()
unique_u = torch.unique(u)
src = unique_u.repeat(num_api)
num_repeat=len(unique_u)
print(num_repeat)
dst = torch.tensor([0+num_mashup]*num_repeat)

for i in range(1,num_api):
    single_dst=torch.tensor([i+num_mashup]*num_repeat)
    dst=torch.cat([dst,single_dst])

all_graph=dgl.graph((src,dst))



# In[10]:


bi_all_cold_graph = to_bidirectional(all_cold_graph)
bi_all_warm_graph = to_bidirectional(all_warm_graph)
bi_all_graph = to_bidirectional(all_graph)


# In[11]:


# 测试：
reverse_train_g=reverse_directional(train_g)
reverse_train_neg_g=reverse_directional(train_neg_g)
reverse_train_pos_g=reverse_directional(train_pos_g)



# ### 构建网络
# 

# In[12]:


"""define model"""

model = Recommender(n_params, args, bi_tag_graph, bi_senmantic_graph)# .to(device)
pred = MLPPredictor(int(args.dim*2))
# pred = MLPPredictor(int(args.dim))
# pred = MLPPredictor(32)

optimizer = torch.optim.Adam(
    itertools.chain(model.parameters(), pred.parameters()), lr=args.lr
)


# In[13]:


# 使用lightGCN
# 警告太麻烦 不让输出了
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tra_adj= get_norm_adj_mat(train_g,num_mashup,num_api,num_nodes)


# In[14]:


dense_adjacency = tra_adj.toarray()


# In[15]:


count=0
for i in range(dense_adjacency.shape[0]):
    if dense_adjacency[i].sum() ==0 :
        count+=1
print(count)


# In[16]:


# ----------- training -------------------------------- #
all_logits = []
for e in range(100):
    # forward
    h,loss_contrast= model(train_g)
    
    # 使用lightGCN
    # h,loss_contrast = model(tra_adj)
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss,bcr_loss= compute_loss(pos_score, neg_score,loss_contrast,h,n_params['n_mashup'],args)

#     # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print("In epoch {}, loss: {}".format(e, loss))

# ----------- check results ------------------------ #
from sklearn.metrics import roc_auc_score

with torch.no_grad():
    # cold:
    pos_score = pred(cold_test_pos_g, h)
    neg_score = pred(cold_test_neg_g, h) 
    score=pred(all_cold_graph,h)

    hit,NDCG ,F1,AUC= evaluation_metrics(all_cold_graph,cold_test_pos_g,score,n_params,pos_score,neg_score,top_k=10)
    print("cold:train loss is {}, val hit@k is {}, val ndcg is {}, val f1 is {}, val auc is {}".format(bcr_loss, hit, NDCG, F1, AUC))

    # warm:
    pos_score = pred(warm_test_pos_g, h)
    neg_score = pred(warm_test_neg_g, h) 
    score=pred(all_warm_graph,h)

    hit,NDCG ,F1,AUC= evaluation_metrics(all_warm_graph,warm_test_pos_g,score,n_params,pos_score,neg_score,top_k=10)
    print("warm:train loss is {}, val hit@k is {}, val ndcg is {}, val f1 is {}, val auc is {}".format(bcr_loss, hit, NDCG, F1, AUC))

    # all:
    pos_score = pred(all_test_pos_g, h)
    neg_score = pred(all_test_neg_g, h) 
    score=pred(all_graph,h)

    hit,NDCG ,F1,AUC= evaluation_metrics(all_graph,all_test_pos_g,score,n_params,pos_score,neg_score,top_k=10)
    print("all:train loss is {}, val hit@k is {}, val ndcg is {}, val f1 is {}, val auc is {}".format(bcr_loss, hit, NDCG, F1, AUC))


# In[17]:


# torch.save(h, 'tensor.pt')

