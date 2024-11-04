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
from custom_GAT import *
import logging
from sklearn.metrics import precision_score,recall_score,ndcg_score



def convert_sp_mat_to_sp_tensor(adj_mat):
    coo = adj_mat.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)


# 为lightGCN构建矩阵，为了加速卷积速度，提前计算邻接矩阵
def get_norm_adj_mat(g,n_mashup,n_api,n_nodes):
    # 创建一个 空的稀疏矩 ，并将其转换为lil_matrix 类型
    adjacency_matrix = sp.dok_matrix((n_nodes, n_nodes), dtype=np.float32)
    adjacency_matrix = adjacency_matrix.tolil()
    # 根据g中的边计算邻接矩阵
    u,v=g.edges()
    R= sp.dok_matrix((n_mashup, n_api), dtype=np.float32)

    for row, col in zip(u, v):
        R[row, col-n_mashup] = 1
        
    '''
        [ 0  R]
        [R.T 0]
    '''
    adjacency_matrix[:n_mashup, n_mashup:] = R
    adjacency_matrix[n_mashup:, :n_mashup] = R.T
    adjacency_matrix = adjacency_matrix.todok()
 
    row_sum = np.array(adjacency_matrix.sum(axis=1))
     
    d_inv = np.power(row_sum, -0.5).flatten()

    d_inv[np.isinf(d_inv)] = 0.
    degree_matrix = sp.diags(d_inv)

    # D^(-1/2) A D^(-1/2)
    norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
    # 为了后续的计算，直接将其转换为tensor类型的稀疏矩阵
    # norm_adjacency= convert_sp_mat_to_sp_tensor(norm_adjacency)

    return norm_adjacency



# 先暂时使用SAGEConv 卷积网络进行 邻居集合 ---后期改为 lightGCN
from dgl.nn import SAGEConv

# ----------------计算的是节点表示----------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        #GraphConv(输入shape，输出shape,聚合函数的类型，...)
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    

class Recommender(nn.Module):
    
    def __init__(self, data_config, args_config, t_graph, s_graph):
        super(Recommender, self).__init__()

        self.n_mashup = data_config['n_mashup']
        self.n_api = data_config['n_api']
        self.n_nodes = data_config['n_nodes'] 
        self.t_graph=t_graph
        self.s_graph=s_graph
        # self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
        #                                                               else torch.device("cpu")
        # args_config.dim = 64
        self.emb_size = args_config.dim
        # local_emd:
        self.local_emb=int(self.emb_size/2)
        # lightGCN的层数
        self.lightgcn_layer = 2
        
        # 计算对比损失时的参数  local
        self.fc1 = nn.Sequential(
                nn.Linear(self.local_emb, self.local_emb, bias=True),
                nn.ReLU(),
                nn.Linear(self.local_emb, self.local_emb, bias=True),
                )
        self.fc2 = nn.Sequential(
                nn.Linear(self.local_emb, self.local_emb, bias=True),
                nn.ReLU(),
                nn.Linear(self.local_emb, self.local_emb, bias=True),
                )
        # global
        self.fc3 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )
        self.fc4 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )

        # 初始化，nodes节点的feat表示：
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))

        # initializer = nn.init.xavier_uniform_
        # self.mashup_embed = initializer(torch.empty(self.n_mashup, self.emb_size))
        
        # initializer = nn.init.xavier_uniform_
        # self.api_embed = initializer(torch.empty(self.n_api, self.emb_size))

        # self.mashup_embed = nn.Parameter(torch.empty(self.n_mashup, self.emb_size))  # 初始化为空张量
        # nn.init.uniform_(self.mashup_embed, -1, 1)  # 使用均匀分布初始化参数

        # self.api_embed = nn.Parameter(torch.empty(self.n_api, self.emb_size))  # 初始化为空张量
        # nn.init.uniform_(self.api_embed, -1, 1)  # 使用均匀分布初始化参数
    
        # 初始化，图卷积网络：
        self.invoke_GraphSAGE = GraphSAGE(self.emb_size, self.emb_size)
        self.tag_GraphSAGE = GraphSAGE(self.emb_size, self.local_emb)
    
        # # self.tag_GAT = GAT(self.t_graph,in_dim=self.emb_size,hidden_dim=8, out_dim=self.local_emb,num_heads=1)

        self.senmantic_GraphSAGE = GraphSAGE(self.emb_size, self.local_emb)

 
    
    # 计算两个输入向量之间的余弦相似度 ---输入向量归一化处理
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    
    def cross_modal_m_loss(self,A_embedding,B_embedding):
        tau = 0.6    # default = 0.8
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc1(A_embedding)
        B_embedding = self.fc1(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret 
    
    def cross_modal_a_loss(self,A_embedding,B_embedding):
        tau = 0.6    # default = 0.8
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc2(A_embedding)
        B_embedding = self.fc2(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret
    
    def cross_view_m_loss(self,A_embedding,B_embedding):
        tau = 0.6    # default = 0.8
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc3(A_embedding)
        B_embedding = self.fc3(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret
    
    def cross_view_a_loss(self,A_embedding,B_embedding):
        tau = 0.6    # default = 0.8
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc4(A_embedding)
        B_embedding = self.fc4(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret 
    
    def light_gcn(self, user_embedding, item_embedding, adj):
        ego_embeddings = torch.cat((user_embedding, item_embedding), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.lightgcn_layer):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            # 累加每一层
            all_embeddings += [ego_embeddings]
        # 将tensor进行堆叠，dim=1 按照 列 进行堆叠  
        all_embeddings = torch.stack(all_embeddings, dim=1)
        # dim=1 在列维度上对emd进行平均，keepdim=False 表示不保留输出张量的维度
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_mashup, self.n_api], dim=0)

        return u_g_embeddings, i_g_embeddings

    # def forward(self,adj):
    def forward(self,g):    
        mashup_emb=self.all_embed[:self.n_mashup,:]
        api_emb=self.all_embed[self.n_mashup:,:]

        # invoke_emd
        invoke_emb=self.invoke_GraphSAGE(g,self.all_embed)
        invoke_m_emb=invoke_emb[:self.n_mashup,:]
        invoke_a_emb=invoke_emb[self.n_mashup:,:]

        # 使用lightGCN计算
        # invoke_m_emb,invoke_a_emb = self.light_gcn(mashup_emb,api_emb,adj)


        # tag_emb
        tag_emb=self.tag_GraphSAGE(self.t_graph,self.all_embed)
        # tag_emb=self.tag_GAT(self.all_embed)
        tag_m_emb=tag_emb[:self.n_mashup,:]
        tag_a_emb=tag_emb[self.n_mashup:,:]

        # #senmantic_emd
        senmantic_emb=self.senmantic_GraphSAGE(self.s_graph,self.all_embed)
        senmantic_m_emb=senmantic_emb[:self.n_mashup,:]
        senmantic_a_emb=senmantic_emb[self.n_mashup:,:]

        # #跨模态对比学习
        m_loss1=self.cross_modal_m_loss(tag_m_emb, senmantic_m_emb)
        a_loss1=self.cross_modal_a_loss(tag_a_emb, senmantic_a_emb)

        # # # 按行拼接，进行全局对比学习
        m_emd1=torch.cat([tag_m_emb,senmantic_m_emb],dim=-1)
        a_emd1=torch.cat([tag_a_emb,senmantic_a_emb],dim=-1)

        # function_embedding=torch.cat([m_emd1,a_emd1],dim=0)

        #跨视图对比学习
        m_loss2=self.cross_view_m_loss(m_emd1, invoke_m_emb)
        a_loss2=self.cross_view_a_loss(a_emd1, invoke_a_emb)
        loss_contrast=m_loss1+a_loss1+m_loss2+a_loss2

        m_emd1=torch.cat([m_emd1,invoke_m_emb],dim=-1)
        a_emd1=torch.cat([a_emd1,invoke_a_emb],dim=-1)
        node_emd=torch.cat([m_emd1,a_emd1],dim=0)

        # # node_emd=torch.cat([m_emd1,a_emd1],dim=0)
        # loss_contrast=m_loss1+a_loss1

        return node_emd, 0.4*(m_loss1+a_loss1)+0.6*(m_loss2+a_loss2)
        # return node_emd, loss_contrast
        

        # return node_emd, loss_contrast,invoke_emb,function_embedding
        # return invoke_emb, 0
        # return function_embedding, m_loss1+a_loss1
        # return senmantic_emb, 0
        # return tag_emb,0



def compute_loss(pos_score, neg_score,loss_contrast,h,n_mashup,data_args):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    decay=data_args.l2
    # bpr
    criteria = nn.BCEWithLogitsLoss()
    bce_loss = criteria(scores, labels)

    mashup_emb=h[:n_mashup,:]
    api_emb=h[n_mashup:,:]
    # L2范数的平方和
    regularizer = (torch.norm(mashup_emb) ** 2
                    + torch.norm(api_emb) ** 2) / 2
    emb_loss = decay * regularizer / n_mashup
    # 对比实验修改  defalu =0.001
    # return bce_loss+0.001*loss_contrast+emb_loss , bce_loss+emb_loss
    return bce_loss+0.01*loss_contrast+emb_loss , bce_loss+emb_loss

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)


def top_k_list(g,score,n_mashup,n_api,top_k=10):
    matrix=np.zeros((n_mashup,n_api))
    src,dst=g.edges()

    for i in range(len(src)):
        matrix[src[i].item(),dst[i].item()-n_mashup]=score[i].item()
    
    top_k_index=np.argpartition(matrix, -top_k, axis=1)[:, -top_k:]
        
    return top_k_index

def NDCG_k(g,pos_g,score,data_config,top_k):
   
    n_mashup = data_config['n_mashup']
    n_api = data_config['n_api']

    src1,dst1=pos_g.edges()
    label_matrix=np.zeros((n_mashup,n_api))

    for i in range(len(src1)):
        label_matrix[src1[i].item(),dst1[i].item()-n_mashup]=1

    src2,dst2=g.edges()
    score_matrix=np.zeros((n_mashup,n_api))
    for i in range(len(src2)):
        if score[i].item()<0:
            score_matrix[src2[i].item(),dst2[i].item()-n_mashup]=0
        else:
            score_matrix[src2[i].item(),dst2[i].item()-n_mashup]=score[i].item()

    ndcg=ndcg_score(label_matrix,score_matrix,k=top_k)
    return ndcg


def NDCG_k2(g,pos_g,score,data_config,top_k):

    src1,dst1 = pos_g.edges()
    unique_u = torch.unique(src1)
    unique_u=unique_u.tolist()
    mapped_u = {value: index for index, value in enumerate(unique_u)}

    n_mashup = len(unique_u)
    all_mashup = data_config['n_mashup']
    n_api = data_config['n_api']

    label_matrix=np.zeros((n_mashup,n_api))

    for i in range(len(src1)):
        label_matrix[mapped_u[src1[i].item()],dst1[i].item()-all_mashup]=1

    src2,dst2=g.edges()

    score_matrix=np.zeros((n_mashup,n_api))
    for i in range(len(src2)):
        if score[i].item()<0:
            score_matrix[mapped_u[src2[i].item()],dst2[i].item()-all_mashup]=0
        else:
            score_matrix[mapped_u[src2[i].item()],dst2[i].item()-all_mashup]=score[i].item()

    ndcg=ndcg_score(label_matrix,score_matrix,k=top_k)
    return ndcg


def hit_k(g,pos_g,score,data_config,top_k):

    n_mashup = data_config['n_mashup']
    n_api = data_config['n_api']

    top_k_index = top_k_list(g,score,n_mashup,n_api,top_k)
    
    src,dst=pos_g.edges()
    matrix=np.zeros((n_mashup,n_api))
    for i in range(len(src)):
        matrix[src[i].item(),dst[i].item()-n_mashup]=1
    
    # 计算hit@k
    R_rate=0    
    n=0
    for i in range(top_k_index.shape[0]):
        hit_count=0
        for j in top_k_index[i]:
            if matrix[i][j]==1:
                hit_count+=1
        if np.sum(matrix[i])!=0:
            n+=1
            R_rate+=hit_count/np.sum(matrix[i])
    hit = R_rate/n
    print(n)
    return hit

def compute_f1(pos_score,neg_score):

    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    # >0.5 则预测正确
    scores=[1 if i>=0.0 else 0 for i in scores]
    f1=f1_score(labels, scores) 

    # precision=precision_score(labels, scores) 
    # recall=recall_score(labels, scores)
    # if (precision+recall)!=0:
    #     f1=2*precision*recall/(precision+recall)
    # else:
    #     f1=0
   
    return f1
    

def evaluation_metrics(g,pos_g,score,data_config,pos_score, neg_score,top_k):

    hit=hit_k(g,pos_g,score,data_config,top_k)
    NDCG=NDCG_k2(g,pos_g,score,data_config,top_k)

    F1=compute_f1(pos_score,neg_score)
    AUC=compute_auc(pos_score, neg_score)

    return hit , NDCG ,F1,AUC




class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        为每个边产生一个标量分数.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        
        # 这里为什么要使用squeeze(1)?----消除不必要的维度
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]
        
