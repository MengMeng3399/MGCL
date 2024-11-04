#!/usr/bin/env python
# coding: utf-8

# <font face="宋体" size=3>
# 
# ## multi_level contrastive Learning ---2024.4.3
# 
# <font color=FF0077>生成各种子图</font>
# 
# 计算构建两个空间的视图：
# -  调用空间；
# -  互补空间：标签共现表征互补关系；语义空间表征的互补关系；
# 
# <font color=00ff77>其中：</font>
# 
# 标签共现互补关系考虑：mashup-api  api-api（去除替补关系）
# 
# 语义空间互补关系考虑：mashup与api描述文档的相似度
# 
# </font>
# 

# In[1]:


import sys
sys.argv = ['run.py']

import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from time import time
from prettytable import PrettyTable
import logging
from utils.parser import parse_args
import dgl
import dgl.data
from utils.data_loader import *

import logging


# In[2]:


"""fix the random seed"""
seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""read args"""
global args, device
args = parse_args()
device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")


# ### 1.加载数据,得到子图

# In[3]:


#  final_data
data_file='/final_data/HGA/invoke_edge'
cold_ratio=0.2
train_g, train_pos_g,train_neg_g,warm_test_pos_g,warm_test_neg_g,cold_test_pos_g,cold_test_neg_g,all_test_pos_g,all_test_neg_g ,all_nodes,number_mashup,g =load_data(args,data_file,cold_ratio)


# In[4]:


print(train_g, train_pos_g,train_neg_g,warm_test_pos_g,warm_test_neg_g,cold_test_pos_g,cold_test_neg_g,all_test_pos_g,all_test_neg_g ,all_nodes,number_mashup,g)


# In[5]:


u,v=cold_test_pos_g.edges()
pos_cold_unique_u = torch.unique(u)
cold_num=len(pos_cold_unique_u)

u,v=warm_test_pos_g.edges()
pos_warm_unique_u = torch.unique(u)
warm_num=len(pos_warm_unique_u)


u,v=all_test_pos_g.edges()
pos_all_unique_u = torch.unique(u)
all_num=len(pos_all_unique_u)

u,v=train_pos_g.edges()
pos_train_unique_u = torch.unique(u)
train_num=len(pos_train_unique_u)

#-------------------

u,v=cold_test_neg_g.edges()
cold_unique_u = torch.unique(u)
cold_num=len(cold_unique_u)

u,v=warm_test_neg_g.edges()
warm_unique_u = torch.unique(u)
warm_num=len(warm_unique_u)


u,v=all_test_neg_g.edges()
all_unique_u = torch.unique(u)
all_num=len(all_unique_u)

u,v=train_neg_g.edges()
train_unique_u = torch.unique(u)
train_num=len(all_unique_u)


# In[6]:


# e = set (pos_cold_unique_u.tolist())
# f = set (pos_all_unique_u.tolist())
# g = set (pos_warm_unique_u.tolist())
# h =set (pos_train_unique_u.tolist())

# a = set (cold_unique_u.tolist())
# b = set (all_unique_u.tolist())
# c = set (warm_unique_u.tolist())
# d =set (train_unique_u.tolist())

# intersection1 = a.intersection(e)
# intersection2 = b.intersection(f)
# intersection3 = c.intersection(g)
# intersection4 = d.intersection(h)

# intersection5 = a.intersection(d)

# print(len(intersection1),len(intersection2),len(intersection3),len(intersection4),len(intersection5))


# In[7]:


def create_tuples(list1, list2):
    # 确保两个列表长度相同
    if len(list1) != len(list2):
        return None

    # 遍历列表，构建元组
    tuples = []
    for i in range(len(list1)):
        tuples.append((list1[i], list2[i]))

    return tuples

# # 例子
# list1 = [1, 2, 3, 4]
# list2 = ['a', 'b', 'c', 'd']
# result = create_tuples(list1, list2)
# print(result)


# In[8]:


# u1,v1=cold_test_neg_g.edges()
# u2,v2=cold_test_pos_g.edges()
# tuples1=create_tuples(u1.tolist(),v1.tolist())
# tuples2=create_tuples(u2.tolist(),v2.tolist())

# set1 = set(tuples1)
# set2 = set(tuples2)
# common_tuples = set1.intersection(set2)

# print("相同的元组：", common_tuples)




# ### 2.得到初始化的 标签互补子图 与 语义互补子图

# In[9]:


data_source='/final_data/HGA'
tag_graph,a_tag_graph,m_tag_graph,tag_matrix,m_tag_dict,a_tag_dict,m_all_remap,a_all_remap=tag_subgraph(args,train_g,number_mashup,all_nodes,data_source)


# In[10]:


len(m_all_remap)


# In[11]:


tag_matrix.shape


# In[12]:


senmantic_graph,bert_emd=semantic_graph(args,number_mashup,all_nodes,data_source)
# bert_emd=semantic_graph(args,number_mashup,all_nodes)


# In[13]:


bert_emd.shape


# In[14]:


all_nodes,number_mashup


# ### 对子图进行保存

# In[15]:


from dgl.data.utils import save_graphs

graph_labels = {"glabel": torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12])}
save_graphs("./data/final_data/HGA/graph.bin", [train_g,train_pos_g,train_neg_g,warm_test_pos_g,warm_test_neg_g,cold_test_pos_g,cold_test_neg_g,all_test_pos_g,all_test_neg_g,tag_graph,senmantic_graph,a_tag_graph,m_tag_graph], graph_labels)



# In[16]:


def to_bidirectional(g):
    src,dst=g.edges()
    new_src=torch.cat([src,dst])
    new_dst=torch.cat([dst,src])
    bidirectional_g = dgl.graph((new_src, new_dst), num_nodes=g.num_nodes())
    return bidirectional_g



# In[17]:


bi_train_g = to_bidirectional(train_g)
bi_train_pos_g = to_bidirectional(train_pos_g)
bi_train_neg_g = to_bidirectional(train_neg_g)
bi_warm_test_pos_g = to_bidirectional(warm_test_pos_g)
bi_warm_test_neg_g = to_bidirectional(warm_test_neg_g)
bi_cold_test_pos_g = to_bidirectional(cold_test_pos_g)
bi_cold_test_neg_g = to_bidirectional(cold_test_neg_g)
bi_all_test_pos_g = to_bidirectional(all_test_pos_g)
bi_all_test_neg_g = to_bidirectional(all_test_neg_g)
bi_tag_graph = to_bidirectional(tag_graph)
bi_senmantic_graph = to_bidirectional(senmantic_graph)
bi_a_tag_graph = to_bidirectional(a_tag_graph)
bi_m_tag_graph = to_bidirectional(m_tag_graph)


# In[18]:


from dgl.data.utils import save_graphs

graph_labels = {"glabel": torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12])}
save_graphs("./data/final_data/HGA/bi_graph.bin", [bi_train_g,bi_train_pos_g,bi_train_neg_g,bi_warm_test_pos_g,bi_warm_test_neg_g,bi_cold_test_pos_g,bi_cold_test_neg_g,bi_all_test_pos_g,bi_all_test_neg_g,bi_tag_graph,bi_senmantic_graph,bi_a_tag_graph,bi_m_tag_graph], graph_labels)


# In[19]:


print(bi_train_g,bi_train_pos_g,bi_train_neg_g,bi_warm_test_pos_g,bi_warm_test_neg_g,bi_cold_test_pos_g,bi_cold_test_neg_g,bi_all_test_pos_g,bi_all_test_neg_g)


# # 对bert_encode 进行保存

# In[21]:


torch.save(bert_emd, './data/final_data/HGA/bert_emd.pt')


# 2.可视化---一个mashup或者一个api通常有多个标签，选择一个作为主标签

# In[ ]:


num_api=all_nodes-number_mashup
data_source='/final_data'
mashup_files= args.data_path + data_source + '/mashup_data' + '.csv'
api_files= args.data_path + data_source + '/api_data'+ '.csv'


# In[ ]:


import operator

def Classification(number,data_file,the_type='mashup',feat="MashupCategory"):
    # 去重,但是不改变顺序
    data= pd.read_csv(data_file)
    ID=[i for i in range(number)]
  
    if the_type == "mashup":
        feat = "MashupCategory"
        
    elif the_type == 'api':
        feat = 'ApiTags'
    else:
        raise ValueError("Error: Type must be 'mashup' or 'api'")
    
    # 由于csv 将list转换为了字符串，因此需要将其改为list
    data[feat]=data[feat].apply(eval)
    temp_list=data.loc[ID,feat].tolist()
    tag_number={}
    
    for sublist in temp_list:
        for item in sublist:
            if item not in tag_number:
                tag_number[item]=1
            else:
                tag_number[item]+=1

    top_tag = sorted(tag_number.items(), key=operator.itemgetter(1), reverse=True)[:15]
    top_tag = {key: value for key, value in top_tag}

    result_dict = {key: [] for key in top_tag.keys()}

    # 如果一个mashup具有多个热门标签，就使用第一个
    for key, value in enumerate(temp_list):
        count = sum(1 for item in value if item in top_tag)
        if count ==1:
            unique_element = next(item for item in value if item in top_tag)
            result_dict[unique_element].append(key)
    
    final_dict={}
    for key,value in result_dict.items():
        if len(value) >= 10:
            final_dict[key]= random.sample(value, 10)

    # 
    
    return tag_number,final_dict,top_tag

    # result_dict={value: index for index, value in enumerate(tag_list)}
    # category_list=[[result_dict[element] for element in row] for row in temp_list]
    # return category_list


# In[ ]:


# mashup_category=Classification(number_mashup,mashup_files)
# api_category=Classification(num_api,api_files,type='api')
# import pickle
# with open('mashup_category', 'wb') as f:
#     pickle.dump(mashup_category, f)

# with open('api_category', 'wb') as f:
#     pickle.dump(api_category, f)


# In[ ]:


M_tag_number,mashup_category,MTOP_category=Classification(number_mashup,mashup_files)
A_tag_number,api_category,ATOP_category=Classification(num_api,api_files,the_type='api')


# In[ ]:


for key,value in M_tag_number.items():
        print(key,value)


# In[ ]:


for key,value in A_tag_number.items():
        print(key,value)


# In[ ]:


MTOP_category


# In[ ]:


# with open('mashup_category.json', 'w') as f:
#     json.dump(mashup_category, f)

# with open('api_category.json', 'w') as f:
#     json.dump(api_category, f)


# 3.introduction的图

# In[ ]:


num_api=all_nodes-number_mashup
data_source='/final_data'
mashup_files= args.data_path + data_source + '/mashup_data' + '.csv'
api_files= args.data_path + data_source + '/api_data'+ '.csv'


# In[ ]:


tag_matrix,m_tag_dict,a_tag_dict,m_all_remap,a_all_remap


# In[ ]:


m_tag=[]
a_tag=[]
for key,value in m_tag_dict.items():
    m_tag.append(key)

for key,value in a_tag_dict.items():
    a_tag.append(key)


# In[ ]:


print(len(m_tag),len(a_tag))


# In[ ]:


only_m_tag=[]
for i in m_tag:
    if i not in a_tag:
        only_m_tag.append(i)


# In[ ]:


len(only_m_tag)


# In[ ]:


for i in only_m_tag:
    print(i,M_tag_number[i])


# In[ ]:


m_tag_dict["Animals"]


# In[ ]:


mashup_data= pd.read_csv("data/final_data/mashup_data.csv")

api_data= pd.read_csv("data/final_data/api_data.csv")


# In[ ]:


a_tag_dict["Media"]


# In[ ]:


for key, value in a_all_remap.items():
    if 74 in value:
        print(api_data['ApiName'][key])


# In[ ]:


for key, value in a_all_remap.items():
    if 59 in value:
        print(api_data['ApiName'][key])


# In[ ]:


for key, value in a_all_remap.items():
    if 59 in value and 58 not in value:
        print(api_data['ApiName'][key])


# In[ ]:


src,dst=g.edges()


# In[ ]:


len(src)


# In[ ]:


youtube_mashup=[]
youtube_mashup_ID=[]

BBC_mashup=[]
BBC_mashup_ID=[]
for i in range(len(dst)):
    # print (dst[i].item()-number_mashup)
    if dst[i].item()== number_mashup+47:
        youtube_mashup.append(mashup_data['MashupName'][src[i].item()])
        youtube_mashup_ID.append(src[i].item())

    if dst[i].item()== number_mashup+197:
        BBC_mashup.append(mashup_data['MashupName'][src[i].item()])
        BBC_mashup_ID.append(src[i].item())


# In[ ]:


youtube_mashup_tag=[]
BBC_mashup_tag=[]

for key,value in m_all_remap.items():
    if key in youtube_mashup_ID:
        youtube_mashup_tag+=value
        
    if key in BBC_mashup_ID:
        BBC_mashup_tag+=value


# In[ ]:


duplicates = list(set(youtube_mashup_tag).intersection(set(BBC_mashup_tag)))


# In[ ]:


duplicates


# In[ ]:


# key=[]
for key,value in m_tag_dict.items():
    if value in duplicates:
        print(key,value)


# In[ ]:


for key,value in m_all_remap.items():
    if key in youtube_mashup_ID and 68 in value:
        print("youtube",mashup_data['MashupName'][key])
        
    if key in BBC_mashup_ID and 68 in value:
        print("BBC",mashup_data['MashupName'][key])
    


# # ---

# In[ ]:


a_tag_dict['Mapping']


# In[ ]:


for key, value in a_all_remap.items():
    if 23 not in value and 156 in value:
        print(api_data["ApiName"][key])


# In[ ]:


src,dst=g.edges()

googlemap_mashup=[]
googlemap_mashup_ID=[]

for i in range(len(dst)):
    # print (dst[i].item()-number_mashup)
    if dst[i].item()== number_mashup+12:
        googlemap_mashup.append(mashup_data['MashupName'][src[i].item()])
        googlemap_mashup_ID.append(src[i].item())


# In[ ]:


m_tag_dict['Travel']


# In[ ]:


for key,value in m_all_remap.items():
    if 32 in value:
        if 'Weather Channel' in mashup_data['MashupRelatedAPIs'][key] and 'Google Maps' in mashup_data['MashupRelatedAPIs'][key]:
            
            print(mashup_data['MashupName'][key])


# In[ ]:


for key,value in m_all_remap.items():
    if 32 in value:
        if 'Weather Channel' in mashup_data['MashupRelatedAPIs'][key]:
            print(mashup_data['MashupName'][key])


# In[ ]:


m_tag_dict['Music']


# In[ ]:


the_key_id=[]
for key,value in m_all_remap.items():
    if 32  not in value and 68 not in value and 132 in value:
        if 'Google Maps' in mashup_data['MashupRelatedAPIs'][key] and 'YouTube' in mashup_data['MashupRelatedAPIs'][key]:
            print(mashup_data['MashupName'][key])
            the_key_id.append(key)


# In[ ]:


m_tag_dict['Travel']


# In[ ]:


a_tag_dict


# In[ ]:


temp_list={}
for i in range(tag_matrix.shape[0]):
    if tag_matrix[31][i]>0:
        temp_list[i]=tag_matrix[31][i]


# In[ ]:





# In[ ]:


temp_list


# In[ ]:


src,dst=g.edges()


# In[ ]:


for i in range(len(src)):
    if 'Travel' in mashup_data['MashupCategory'][src[i].item()] and 'Mapping' in api_data['ApiTags'][dst[i].item()-number_mashup]:
        print(mashup_data['MashupName'][src[i].item()],"..", api_data['ApiName'][dst[i].item()-number_mashup])


# In[ ]:


temp_list=[]
src,dst=senmantic_graph.edges()
for i in range(len(src)):
    if src[i].item() == 18 :
        temp_list.append(dst[i].item()-number_mashup)


# In[ ]:


for i in temp_list:
    print(api_data['ApiName'][i])

