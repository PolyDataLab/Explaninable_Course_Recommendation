#sophos server
# adding one user at a time
import time
import pandas as pd
from preprocess_v4_dser import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
#from topic_model_v2 import *
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import Linear, ModuleList
from torch_geometric.utils import softmax
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from torch_geometric.utils import degree, dropout_adj
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import GATConv

def measure_sequence_of_courses(data, reversed_item_dict):
    #users = data.userID.values
    #print(users)
    #time_baskets = data.baskets.values
    # item_list = []
    # for baskets in data['baskets']:
    #     for basket in baskets:
    #         for item in basket:
    #             if item not in item_list:
    #                 item_list.append(item)
    num_items = len(reversed_item_dict)
    #sequence_dict = {}
    #count_item= {}
    #index1= 0
    seq_matrix = np.zeros((num_items, num_items))
    for baskets in data['baskets']:
        for index1 in range(0, len(baskets)-1):
            for index2 in range(index1+1, len(baskets)):
                list1= baskets[index1]
                list2= baskets[index2]
                for item1 in list1:
                    for item2 in list2:
                        #sequence_dict[item1, item2]= sequence_dict.get((item1, item2),0)+ 1 
                        seq_matrix[reversed_item_dict[item1]][reversed_item_dict[item2]] += 1

    seq_matrix = normalize(seq_matrix, norm='l2') 
    return seq_matrix

def compute_tfidf_pmi(documents, window_size=10):
    """
    Computes the adjacency matrix A combining TF-IDF and PMI.
    
    Parameters:
    - documents: list of str, corpus of documents.
    - window_size: int, context window size for PMI calculation.
    
    Returns:
    - adjacency_matrix: scipy.sparse.csr_matrix, combined adjacency matrix.
    """
    # Step 1: Compute TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)  # Shape: (num_docs, vocab_size)
    vocab = vectorizer.get_feature_names_out()
    vocab_size = len(vocab)
    vocab_dict_idx_to_wrd = {i: word for i, word in enumerate(vocab)}
    vocab_dict_wrd_to_idx = {word: i for i, word in enumerate(vocab)}
    #print(vocab_size)
    
    # Step 2: PMI Calculation
    # Build a word co-occurrence matrix
    word_cooccurrence = np.zeros((vocab_size, vocab_size))
    
    for doc in documents:
        tokens = doc.split()
        for i, token in enumerate(tokens):
            if token not in vocab:
                continue
            token_idx = np.where(vocab == token)[0][0]
            for j in range(max(i - window_size, 0), min(i + window_size + 1, len(tokens))):
                if i == j or tokens[j] not in vocab:
                    continue
                context_idx = np.where(vocab == tokens[j])[0][0]
                word_cooccurrence[token_idx, context_idx] += 1
    
    # Normalize co-occurrence counts to probabilities
    word_sums = word_cooccurrence.sum(axis=1)
    total_sum = word_cooccurrence.sum()
    pmi_matrix = np.zeros_like(word_cooccurrence)
    
    for i in range(vocab_size):
        for j in range(vocab_size):
            if word_cooccurrence[i, j] > 0:
                p_ij = word_cooccurrence[i, j] / total_sum
                p_i = word_sums[i] / total_sum
                p_j = word_sums[j] / total_sum
                pmi_matrix[i, j] = max(0, np.log(p_ij / (p_i * p_j)))  # PMI formula
    
    # # Step 3: Combine TF-IDF and PMI into adjacency matrix
    # tfidf_csr = tfidf_matrix.T  # (vocab_size, num_docs)
    # adjacency_matrix = csr_matrix((vocab_size + tfidf_matrix.shape[0], vocab_size + tfidf_matrix.shape[0]))
    
    # # Fill in document-word (TF-IDF) and word-word (PMI)
    # adjacency_matrix[:vocab_size, vocab_size:] = tfidf_csr
    # adjacency_matrix[vocab_size:, :vocab_size] = tfidf_csr.T
    # adjacency_matrix[:vocab_size, :vocab_size] = csr_matrix(pmi_matrix)
    # tfidf_csr = tfidf_matrix  # (num_docs, vocab_size)
    tfidf_csr = csr_matrix(tfidf_matrix)

    # Create an adjacency matrix of the required shape
    adjacency_matrix = csr_matrix((tfidf_matrix.shape[0]+vocab_size, vocab_size))

    # Fill in document-word (TF-IDF)
    adjacency_matrix[:tfidf_matrix.shape[0], :] = tfidf_csr  # Documents to Words (TF-IDF)

    # Fill in word-word (PMI)
    # adjacency_matrix[:vocab_size, :] = csr_matrix(pmi_matrix)  # Words to Words (PMI)
    adjacency_matrix[tfidf_matrix.shape[0]:, :] = csr_matrix(pmi_matrix)  # Words to Words (PMI)

    
    return adjacency_matrix, vocab_size, vocab_dict_idx_to_wrd, vocab_dict_wrd_to_idx

def get_adj_matrix_text(data, window_s, reversed_item_dict_one_hot):

    text_f_all = {}
    for idx1, cid in enumerate(data["course_code"]):
        #index = data[data['userID'] == user].index.values[0]
        idx2 = reversed_item_dict_one_hot[cid] # cid to idx
        # cid2 = data["course_code"][idx2]
        cname2 = data["course_name"][idx2]
        text_f = data["course_description"][idx2]
        row1 = [cname2, text_f]
        text_f_all[idx2] = row1
        #text_f_all[idx2] = text_f

        # cat_f = data["category"][idx1]
        # level_f = data["level"][idx1]
        # row1 = [cid, cat_f]
        # row2 = [cid, str(level_f)]
        # # cat_f_all.append(row1)
        # # level_f_all.append(row2)
        # cat_f_all[idx1] = row1
        # level_f_all[idx1] = row2
    text_f_all_sorted = dict(sorted(text_f_all.items(), key=lambda item: item[0], reverse=False))
    text_f_all_new = list(text_f_all_sorted.values())
    # item_descriptions = data['course_description'].dropna().tolist()
   # item_ids = data['course_code'].dropna().tolist()
    # item_names = data['course_name'].dropna().tolist()
    item_descriptions = []
    item_names = []
    for list1 in text_f_all_new:
        cname3, cdesc = list1
        item_descriptions.append(cdesc)
        item_names.append(cname3)

    for it_desc_idx in range(len(item_descriptions)):
        idx_substring = item_descriptions[it_desc_idx].find(item_names[it_desc_idx])
        if idx_substring ==-1: # course name not found in course description
            updated_it_desc = item_names[it_desc_idx]+ " "+ item_descriptions[it_desc_idx] # adding course name
            item_descriptions[it_desc_idx] = updated_it_desc
   
    adj_mat, vocab_size, vocab_dict_idx_to_wrd, vocab_dict_wrd_to_idx = compute_tfidf_pmi(item_descriptions, window_size=window_s)
    return adj_mat, vocab_size, vocab_dict_idx_to_wrd, vocab_dict_wrd_to_idx

def get_adj_matrix_text_v2(data, reversed_item_dict_one_hot):

    concept_f_all = {}
    for idx1, cid in enumerate(data["course_code"]):
        #index = data[data['userID'] == user].index.values[0]
        idx2 = reversed_item_dict_one_hot[cid] # cid to idx
        # cid2 = data["course_code"][idx2]
        cname2 = data["course_name"][idx2]
        cid2 = data["course_code"][idx2]
        #text_f = data["course_description"][idx2]
        concept_f = data["concepts"][idx2]
        #row1 = [cname2, text_f]
        # row1 = [cname2, concept_f]
        row1 = [cid2, concept_f]
        concept_f_all[idx2] = row1
        #text_f_all[idx2] = text_f

        # cat_f = data["category"][idx1]
        # level_f = data["level"][idx1]
        # row1 = [cid, cat_f]
        # row2 = [cid, str(level_f)]
        # # cat_f_all.append(row1)
        # # level_f_all.append(row2)
        # cat_f_all[idx1] = row1
        # level_f_all[idx1] = row2
    concept_f_all_sorted = dict(sorted(concept_f_all.items(), key=lambda item: item[0], reverse=False))
    concept_f_all_new = list(concept_f_all_sorted.values())
    # item_descriptions = data['course_description'].dropna().tolist()
   # item_ids = data['course_code'].dropna().tolist()
    # item_names = data['course_name'].dropna().tolist()
    # item_descriptions = []
    item_concepts = {}
    #item_names = []
    item_id = 0
    for list1 in concept_f_all_new:
        cid3, cconcept = list1
        item_concepts[cid3] = cconcept
        item_id += 1
        #item_names.append(cname3)
    # Get a sorted list of all unique concepts
    #all_concepts = sorted(set.union(*item_concepts.values()))
    all_concepts = sorted(set.union(*map(set, item_concepts.values())))

    num_items = len(item_concepts)
    num_concepts = len(all_concepts)
    vocab_size = num_concepts

    # Create an adjacency matrix (items x concepts)
    adj_matrix = np.zeros((num_items, num_concepts), dtype=int)

    # Fill the adjacency matrix
    concept_to_idx = {concept: idx for idx, concept in enumerate(all_concepts)}
    idx_to_concept = {idx: concept for idx, concept in enumerate(all_concepts)}

    for item, concepts in item_concepts.items():
        for concept in concepts:
            adj_matrix[reversed_item_dict_one_hot[item], concept_to_idx[concept]] = 1  # Mark connection

    print("Binary Adjacency Matrix (Items x Concepts):")
    print(adj_matrix)

    # Optional: Print concept labels for better readability
    #import pandas as pd
    #df = pd.DataFrame(adj_matrix, index=[f"Item_{i}" for i in range(num_items)], columns=all_concepts)
    #print(df)

   
    #adj_mat, vocab_size = compute_tfidf_pmi(item_concepts, window_size=window_s)
    return adj_matrix, vocab_size, concept_to_idx, idx_to_concept

def get_adj_matrix_text_v3(data, reversed_item_dict_one_hot, window_s):

    concept_f_all = {}
    for idx1, cid in enumerate(data["course_code"]):
        #index = data[data['userID'] == user].index.values[0]
        idx2 = reversed_item_dict_one_hot[cid] # cid to idx
        # cid2 = data["course_code"][idx2]
        cname2 = data["course_name"][idx2]
        cid2 = data["course_code"][idx2]
        #text_f = data["course_description"][idx2]
        concept_f = data["concepts"][idx2]
        #row1 = [cname2, text_f]
        # row1 = [cname2, concept_f]
        row1 = [cid2, concept_f]
        concept_f_all[idx2] = row1
        #text_f_all[idx2] = text_f

        # cat_f = data["category"][idx1]
        # level_f = data["level"][idx1]
        # row1 = [cid, cat_f]
        # row2 = [cid, str(level_f)]
        # # cat_f_all.append(row1)
        # # level_f_all.append(row2)
        # cat_f_all[idx1] = row1
        # level_f_all[idx1] = row2
    concept_f_all_sorted = dict(sorted(concept_f_all.items(), key=lambda item: item[0], reverse=False))
    concept_f_all_new = list(concept_f_all_sorted.values())
    # item_descriptions = data['course_description'].dropna().tolist()
   # item_ids = data['course_code'].dropna().tolist()
    # item_names = data['course_name'].dropna().tolist()
    # item_descriptions = []
    # item_concepts = {}
    # #item_names = []
    # item_id = 0
    # for list1 in concept_f_all_new:
    #     cid3, cconcept = list1
    #     item_concepts[cid3] = cconcept
    #     item_id += 1
    #     #item_names.append(cname3)
    # # Get a sorted list of all unique concepts
    # #all_concepts = sorted(set.union(*item_concepts.values()))
    # all_concepts = sorted(set.union(*map(set, item_concepts.values())))

    # num_items = len(item_concepts)
    # num_concepts = len(all_concepts)
    # vocab_size = num_concepts

    # # Create an adjacency matrix (items x concepts)
    # adj_matrix = np.zeros((num_items, num_concepts), dtype=int)

    # # Fill the adjacency matrix
    # concept_to_idx = {concept: idx for idx, concept in enumerate(all_concepts)}
    # idx_to_concept = {idx: concept for idx, concept in enumerate(all_concepts)}

    # for item, concepts in item_concepts.items():
    #     for concept in concepts:
    #         adj_matrix[reversed_item_dict_one_hot[item], concept_to_idx[concept]] = 1  # Mark connection

    # print("Binary Adjacency Matrix (Items x Concepts):")
    # print(adj_matrix)

    # Optional: Print concept labels for better readability
    #import pandas as pd
    #df = pd.DataFrame(adj_matrix, index=[f"Item_{i}" for i in range(num_items)], columns=all_concepts)
    #print(df)
    item_descriptions = []
    #item_names = []
    for list1 in concept_f_all_new:
        cid3, cconcept = list1
        cdesc = ' '.join(cconcept)
        item_descriptions.append(cdesc)
        #item_names.append(cname3)

    # for it_desc_idx in range(len(item_descriptions)):
    #     idx_substring = item_descriptions[it_desc_idx].find(item_names[it_desc_idx])
    #     if idx_substring ==-1: # course name not found in course description
    #         updated_it_desc = item_names[it_desc_idx]+ " "+ item_descriptions[it_desc_idx] # adding course name
    #         item_descriptions[it_desc_idx] = updated_it_desc
   
    adj_matrix, vocab_size, vocab_dict_idx_to_wrd, vocab_dict_wrd_to_idx = compute_tfidf_pmi(item_descriptions, window_size=window_s)

   
    #adj_mat, vocab_size = compute_tfidf_pmi(item_concepts, window_size=window_s)
    return adj_matrix, vocab_size, vocab_dict_idx_to_wrd, vocab_dict_wrd_to_idx



def convert_to_one_hot_encoding_cat(data):
    # One-hot encode the 'interactions' column
    data['cat_f'] = data['cat_f'].apply(lambda x: [x])
    mlb = MultiLabelBinarizer()
    one_hot_encoded = mlb.fit_transform(data['cat_f'])
    # Convert back to a DataFrame for easier readability
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=mlb.classes_, index=data['item']).reset_index(drop=True)
    cat_dict_one_hot = {index: cat_f for index, cat_f in enumerate(one_hot_df.columns)} # idx , cat
    item_dict_one_hot_cat = {index: item for index, item in enumerate(data['item'])} # idx , cid
    one_hot_df['item'] = data['item']
    list1 =  list(cat_dict_one_hot.values())
    one_hot_df = one_hot_df[['item'] + list1]
    #print(one_hot_df.shape)
    
    return one_hot_encoded, one_hot_df, item_dict_one_hot_cat, cat_dict_one_hot

def convert_to_one_hot_encoding_level(data):
   # One-hot encode the 'interactions' column
    data['level_f'] = data['level_f'].apply(lambda x: [x])
    mlb = MultiLabelBinarizer()
    one_hot_encoded = mlb.fit_transform(data['level_f'])
    # Convert back to a DataFrame for easier readability
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=mlb.classes_, index=data['item']).reset_index(drop=True)
    level_dict_one_hot = {index: level_f for index, level_f in enumerate(one_hot_df.columns)} # idx , level
    item_dict_one_hot_level = {index: item for index, item in enumerate(data['item'])} # idx , cid
    one_hot_df['item'] = data['item']
    list1 =  list(level_dict_one_hot.values())
    one_hot_df = one_hot_df[['item'] + list1]
    #print(one_hot_df.shape)
    
    return one_hot_encoded, one_hot_df, item_dict_one_hot_level, level_dict_one_hot

def convert_side_info_to_one_hot_encoding(data, reversed_item_dict_one_hot, num_items):
    # df["category"] 
    # df["level"] 
    cat_f_all = {}
    level_f_all = {}
    for idx1, cid in enumerate(data["course_code"]):
        #index = data[data['userID'] == user].index.values[0]
        idx2 = reversed_item_dict_one_hot[cid] # cid to idx
        cid2 = data["course_code"][idx2]
        cat_f = data["category"][idx2]
        level_f = data["level"][idx2]
        row1 = [cid2, cat_f]
        row2 = [cid2, str(level_f)]
        cat_f_all[idx2] = row1
        level_f_all[idx2] = row2

        # cat_f = data["category"][idx1]
        # level_f = data["level"][idx1]
        # row1 = [cid, cat_f]
        # row2 = [cid, str(level_f)]
        # # cat_f_all.append(row1)
        # # level_f_all.append(row2)
        # cat_f_all[idx1] = row1
        # level_f_all[idx1] = row2
    cat_f_all_sorted = dict(sorted(cat_f_all.items(), key=lambda item: item[0], reverse=False))
    level_f_all_sorted = dict(sorted(level_f_all.items(), key=lambda item: item[0], reverse=False))
    cat_f_all = list(cat_f_all_sorted.values())
    level_f_all = list(level_f_all_sorted.values())
    
    cat_f_all_df = pd.DataFrame(cat_f_all, columns=['item', 'cat_f'])
    level_f_all_df = pd.DataFrame(level_f_all, columns=['item', 'level_f'])
    one_hot_encoded_cat, one_hot_df_cat, item_dict_one_hot_cat, cat_dict_one_hot = convert_to_one_hot_encoding_cat(cat_f_all_df)
    one_hot_encoded_level, one_hot_df_level, item_dict_one_hot_level, level_dict_one_hot = convert_to_one_hot_encoding_level(level_f_all_df)

    reversed_dict_cat_to_idx = dict(zip(cat_dict_one_hot.values(), cat_dict_one_hot.keys()))  # cat, idx
    reversed_dict_level_to_idx = dict(zip(level_dict_one_hot.values(), level_dict_one_hot.keys()))  # level, idx

    return one_hot_encoded_cat, one_hot_encoded_level, cat_dict_one_hot, level_dict_one_hot, reversed_dict_cat_to_idx,  reversed_dict_level_to_idx, one_hot_df_cat, one_hot_df_level


# def create_embedding_for_training_users(one_hot_encoded_data, num_users_train, item_embeddings, item_dict_idx_to_cid, item_dict_cid_to_idx, reversed_item_dict_one_hot, num_items):
#     # num_nodes = data.x.size(0)
    
#     #num_nodes = final_x.size(0)
#     # Initialize new user features as the mean of the features of interacted items
#     #item_embeddings = final_x[num_users:]
#     interaction_matrix = np.array(one_hot_encoded_data)
#     #num_items = num_nodes // 2  # Assuming first half are users and second half are items
#     item_embeddings2 = np.zeros((item_embeddings.shape[0],item_embeddings.shape[1]))
#     for idx1, it_embed in enumerate(item_embeddings):
#         cid = item_dict_idx_to_cid[idx1]
#         idx2 = reversed_item_dict_one_hot[cid] # cid to idx
#         item_embeddings2[idx2] = it_embed

    
#     #num_users_new = one_hot_encoded_data.shape[0]
#     new_user_interactions_all = []
#     for user in range(num_users_train):
#         new_user_interactions = []
#         for item in range(num_items):
#             if interaction_matrix[user, item] == 1:
#                 new_user_interactions.append(item)
#         new_user_interactions_all.append(new_user_interactions)
    
#    # new_user_interactions2 = []
#    # for item_idx1 in new_user_interactions:
#         #cid = item_dict_one_hot[item_idx1]
#         #new_user_interactions2.append(item_dict_cid_to_idx[cid]) # index of a course in item embeddings we get from topic modeling
#     #new_user_embedding = item_embeddings[new_user_interactions2].mean(dim=0, keepdim=True)
#     new_user_embeddings_all = []
#     for new_user_interactions2 in new_user_interactions_all:
#         # new_user_embedding = np.mean(item_embeddings2[new_user_interactions2], axis=0)[np.newaxis, :]
#         new_user_embedding = np.mean(item_embeddings2[new_user_interactions2], axis=0).flatten()
#         new_user_embeddings_all.append(new_user_embedding)
#     new_user_embeddings2 = np.array(new_user_embeddings_all)

#     # Update node features
#     #updated_x = torch.cat([data.x, new_user_embedding], dim=0)
      
    
#     # Update edge index to include new edges between the new user and interacted items
#     #new_user_index_all = []
#     #new_user_index = torch.tensor([[num_nodes] * len(new_user_interactions),  [item_idx + num_users for item_idx in new_user_interactions]])
#         #new_user_index_all.append(new_user_index)
#     #updated_edge_index = torch.cat([data.edge_index, new_user_index], dim=1)
    
#     # Create new Data object with updated features and edges
#     #updated_data = Data(x=updated_x, edge_index=updated_edge_index)
    
#     return new_user_embeddings2, item_embeddings2

# class LightGCNLayer(MessagePassing):
#     def __init__(self):
#         super(LightGCNLayer, self).__init__(aggr="add")  # 'add' aggregation for GCN

#     def forward(self, x, edge_index):
#         # Normalize node embeddings with degree normalization
#         row, col = edge_index
#         deg = torch.bincount(row, minlength=x.size(0))
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         return self.propagate(edge_index, x=x, norm=norm)

#     def message(self, x_j, norm):
#         # Multiply embeddings by normalized degree
#         return norm.view(-1, 1) * x_j
    
# class LightGCN(torch.nn.Module):
#     def __init__(self, num_users, num_items, num_features, embedding_dim, num_layers, dropout):
#         super(LightGCN, self).__init__()
#         self.num_users = num_users
#         self.num_items = num_items
#         self.num_features = num_features
#         self.embedding_dim = embedding_dim
#         self.num_layers = num_layers
#         self.dropout = torch.nn.Dropout(dropout)
#         # self.user_embeddings_LDA = user_embeddings_LDA
#         # self.item_embeddings_LDA = item_embeddings_LDA

#         # Initialize user and item embeddings
#         self.item_embeddings = torch.nn.Embedding(num_items, embedding_dim)
#         self.user_embeddings = torch.nn.Embedding(num_users, embedding_dim)
#         self.fet_embeddings = torch.nn.Embedding(num_features, embedding_dim)
#         # self.item_embeddings.weight = torch.nn.Parameter(torch.cat([self.item_embeddings.weight, self.item_embeddings_LDA], dim=1))
#         # self.user_embeddings.weight = torch.nn.Parameter(torch.cat([self.user_embeddings.weight, self.user_embeddings_LDA], dim=1))

#         torch.nn.init.xavier_uniform_(self.item_embeddings.weight)
#         torch.nn.init.xavier_uniform_(self.user_embeddings.weight)
#         torch.nn.init.xavier_uniform_(self.fet_embeddings.weight)
#         # torch.nn.init.normal_(self.item_embeddings.weight, std=0.1)
#         # torch.nn.init.normal_(self.user_embeddings.weight, std=0.1)

#         # Create LightGCN layers
#         self.convs = torch.nn.ModuleList([LightGCNLayer() for _ in range(num_layers)])

#     def forward(self, edge_index): 
#     #def forward(self, edge_index):
#         # Concatenate user and item embeddings
#         #if (x.size(0)==(self.num_users+self.num_items)):
#         # x = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
#         x = torch.cat([self.item_embeddings.weight, self.user_embeddings.weight, self.fet_embeddings.weight], dim=0)
        
#         all_embeddings = [x]

#         # Apply each LightGCN layer
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             all_embeddings.append(x)

#         # Average embeddings from each layer
#         final_embedding = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)

#         # Separate final user and item embeddings
#         # user_embedding, item_embedding = final_embedding[:self.num_users], final_embedding[self.num_users:]
#         item_embedding, user_embedding, fet_embedding = final_embedding[:self.num_items], final_embedding[self.num_items:self.num_items+self.num_users], final_embedding[self.num_items+self.num_users:]
#         return item_embedding, user_embedding, fet_embedding, final_embedding
    
#     # # Example forward pass with new test user
#     # def forward_with_test_user(self, updated_x, updated_edge_index):

#     #     # Perform forward pass with updated x and edge_index
#     #     user_embeddings, item_embeddings, final_x = model.forward(updated_edge_index)
#     #     print(final_x.shape)
#     #     return user_embeddings, item_embeddings, final_x
# class GATLayer(MessagePassing):
#     def __init__(self, in_channels, out_channels, heads=1, dropout_prob=0.0):
#         super(GATLayer, self).__init__(aggr="add")  # 'add' aggregation for GAT
#         self.heads = heads
#         self.dropout_prob = dropout_prob
#         self.lin = Linear(in_channels, heads * out_channels, bias=False)
#         self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
#         self.reset_parameters()

#         # Attribute to store attention weights
#         self.edge_attention = None

#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.lin.weight)
#         torch.nn.init.xavier_uniform_(self.att)

#     def forward(self, x, edge_index, edge_weight=None):
#         x = self.lin(x)
#         x = x.view(-1, self.heads, x.size(-1) // self.heads)  # Shape: [num_nodes, heads, out_channels]

#         # Edge dropout (optional)
#         if self.dropout_prob > 0:
#             edge_index, edge_weight = self.dropout_edges(edge_index, edge_weight)

#         # Propagate messages
#         out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

#         # Aggregate attention outputs
#         return out.mean(dim=1)  # Average over heads

#     def message(self, x_i, x_j, edge_index, edge_weight):
#         print(f"x_i.shape: {x_i.shape}, x_j.shape: {x_j.shape}")
#         print(f"edge_index.shape: {edge_index.shape}")
#         edge_attr = torch.cat([x_i, x_j], dim=-1)  # Shape: [num_edges, heads, 2 * out_channels]
#         alpha = (edge_attr * self.att).sum(dim=-1)  # Shape: [num_edges, heads]
#         alpha = F.leaky_relu(alpha, negative_slope=0.2)
#         alpha = softmax(alpha, edge_index[0])  # Normalize across source nodes

#         # Store attention weights for later use
#         self.edge_attention = alpha  # Shape: [num_edges, heads]

#         # Apply edge weights if provided
#         if edge_weight is not None:
#             alpha = alpha * edge_weight.unsqueeze(1)

#         # Dropout on attention scores
#         alpha = F.dropout(alpha, p=self.dropout_prob, training=self.training)

#         return alpha.unsqueeze(-1) * x_j  # Weighted message: Shape [num_edges, heads, out_channels]

#     def dropout_edges(self, edge_index, edge_weight):
#         if edge_weight is not None:
#             mask = torch.rand(edge_weight.size(0)) > self.dropout_prob
#             edge_index = edge_index[:, mask]
#             edge_weight = edge_weight[mask]
#         return edge_index, edge_weight


class GAT(torch.nn.Module):
    def __init__(self, num_users, num_items, num_fet, embedding_dim, num_layers, edge_dropout, node_dropout, num_heads=1, seed_value = 42):
        super(GAT, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_fet = num_fet
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.edge_dropout = edge_dropout
        self.node_dropout = node_dropout
        self.num_heads = num_heads

        torch.manual_seed(seed_value)  # Ensure same random initialization
        # Initialize user and item embeddings
        self.item_embeddings = torch.nn.Embedding(self.num_items, embedding_dim)
        self.user_embeddings = torch.nn.Embedding(self.num_users, embedding_dim)
        self.fet_embeddings = torch.nn.Embedding(self.num_fet, embedding_dim)
        torch.nn.init.xavier_uniform_(self.item_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.user_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.fet_embeddings.weight)
        print("init emb: ", self.item_embeddings.weight[:2])  # Print first 5 embeddings

        # # Define GAT layers
        # self.layers = ModuleList([
        #     GATLayer(embedding_dim, embedding_dim, heads=heads, dropout_prob=self.edge_dropout)
        #     for _ in range(num_layers)
        # ])
        # GAT layers
        in_channels = out_channels = hidden_channels = embedding_dim
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=edge_dropout)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=edge_dropout)
        self.gat3 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=edge_dropout)
        #self.linear_transform = torch.nn.Linear(hidden_channels * num_heads, embedding_dim)

    def forward(self, edge_index, edge_weight=None):
        x = torch.cat([self.item_embeddings.weight, self.user_embeddings.weight, self.fet_embeddings.weight], dim=0)
        all_embeddings = [x]
        attention_weights = []  # List to store attention weights
        edge_index_out_all = []

        if self.num_layers > 1:
            # First GAT layer
            x, (edge_index_out, alpha) = self.gat1(x, edge_index, edge_weight, return_attention_weights=True)
            attention_weights.append(alpha)
            edge_index_out_all.append(edge_index_out)
            all_embeddings.append(x)

            for _ in range(self.num_layers - 2):
                x = F.relu(x)
                x = F.dropout(x, p=self.node_dropout, training=self.training)
                x, (edge_index_out, alpha) = self.gat2(x, edge_index, return_attention_weights=True)
                attention_weights.append(alpha)
                edge_index_out_all.append(edge_index_out)
                all_embeddings.append(x)

            # Final GAT layer
            x = F.relu(x)
            x = F.dropout(x, p=self.node_dropout, training=self.training)
            x, (edge_index_out, alpha) = self.gat3(x, edge_index, return_attention_weights=True)
            attention_weights.append(alpha)
            edge_index_out_all.append(edge_index_out)
            all_embeddings.append(x)

        else:
            # Single-layer GAT
            x, (edge_index_out, alpha) = self.gat1(x, edge_index, edge_weight, return_attention_weights=True)
            attention_weights.append(alpha)
            all_embeddings.append(x)
            edge_index_out_all.append(edge_index_out)

        # Aggregate all embeddings
        if self.num_heads == 1:
            final_embedding = torch.mean(torch.stack(all_embeddings, dim=0), dim=0) # avg over emb of all attention layers
            #final_embedding = x
        else:
            all_emb = [all_embeddings[0]]
            for emb in all_embeddings[1:]:
                if emb.size(-1) != self.embedding_dim:
                    emb = emb.view(emb.size(0), self.num_heads, -1)
                    emb = emb.mean(dim=1)
                all_emb.append(emb)
            final_embedding = torch.mean(torch.stack(all_emb, dim=0), dim=0)  # avg over emb of all attention layers
            #final_embedding = all_emb[-1]
        #final_embedding = x
        # Split final embedding
        item_embedding = final_embedding[:self.num_items]
        user_embedding = final_embedding[self.num_items:self.num_items + self.num_users]
        fet_embedding = final_embedding[self.num_items + self.num_users:]

        return item_embedding, user_embedding, fet_embedding, final_embedding, attention_weights, edge_index_out_all

    
    def bpr_loss(self, user_embeddings, pos_item_embeddings, neg_item_embeddings):
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)
        # pos_f_scores = torch.sum(user_embeddings * pos_f_embeddings, dim=1)
        # neg_f_scores = torch.sum(user_embeddings * neg_f_embeddings, dim=1)
        # return -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))


def train_model(one_hot_encoded_train, one_hot_encoded_cat, one_hot_encoded_level, one_hot_encoded_text, n_layers, embedding_dim, n_epochs, l_rate, edge_dropout, node_dropout, heads, threshold_weight_edges_iw,  threshold_weight_edges_ww, seed_value, version):
    # Step 1: Data Preparation (Assume a small dataset)
    # Simulate user-item interaction graph (0: no interaction, 1: interaction)
    # Rows represent users, columns represent items
    num_users = one_hot_encoded_train.shape[0]
    num_items = one_hot_encoded_train.shape[1]
    num_cat_f = one_hot_encoded_cat.shape[1] 
    num_level_f = one_hot_encoded_level.shape[1]
    num_text_f = one_hot_encoded_text.shape[1]
    # num_fet = one_hot_encoded_cat.shape[1] + one_hot_encoded_level.shape[1]
    num_fet = num_cat_f + num_level_f + num_text_f

    # # Example interaction matrix (4 users, 6 items)
    # interaction_matrix = np.array([
    #     [1, 1, 0, 0, 0, 1],  # User 1 interacted with Item 1, 2, and 6
    #     [0, 0, 1, 1, 0, 0],  # User 2 interacted with Item 3 and 4
    #     [1, 0, 0, 1, 1, 0],  # User 3 interacted with Item 1, 4, 5
    #     [0, 1, 0, 1, 1, 0]   # User 4 interacted with Item 2, 4, 5
    # ])
    # interaction_matrix = np.array(one_hot_encoded_train)
    # Create edges (user-item interactions as bipartite edges)
    # edge_index = []
    # for user in range(num_users):
    #     for item in range(num_items):
    #         if interaction_matrix[user, item] == 1:
    #             # edge_index.append([user, num_users + item])  # User connected to Item
    #              edge_index.append([item, num_items + user]) 
    # interaction_matrix_cat = np.array(one_hot_encoded_cat)
    # for item in range(num_items):
    #     for cat in range(num_cat_f):
    #         if interaction_matrix_cat[item, cat] == 1:
    #             # edge_index.append([user, num_users + item])  # User connected to Item
    #              edge_index.append([item, (num_items + num_users+ cat)]) 
    
    # interaction_matrix_level = np.array(one_hot_encoded_level)
    # for item in range(num_items):
    #     for level in range(num_level_f):
    #         if interaction_matrix_level[item, level] == 1:
    #             # edge_index.append([user, num_users + item])  # User connected to Item
    #              edge_index.append([item, (num_items + num_users+ num_cat_f+level)]) 


    # edge_index = torch.tensor(edge_index).t().contiguous()
    interaction_matrix = np.array(one_hot_encoded_train)
    adj_matrix_all = np.zeros((num_items+num_users+num_fet, num_items+num_users+num_fet))
    edge_index = []
    # user item edges
    cnt4 = 0
    for user in range(num_users):
        for item in range(num_items):
            if interaction_matrix[user, item] == 1:
                 cnt4+= 1
                # edge_index.append([user, num_users + item])  # User connected to Item
                 edge_index.append([item, num_items + user]) 
                 adj_matrix_all[item, num_items + user] = 1
                 #adj_matrix_all[num_items + user, item] = 1
    cnt1= 0
    #item item sequential edges
    # interaction_matrix_cc = np.array(cc_seq_matrix)
    # for item1 in range(num_items):
    #     for item2 in range(num_items):
    #         if interaction_matrix_cc[item1, item2] >= threshold_weight_edges_cc: # threshold, th = 0.1
    #              cnt1+= 1
    #              cnt4+= 1
    #             # edge_index.append([user, num_users + item])  # User connected to Item
    #              edge_index.append([item1, item2]) 
    #              adj_matrix_all[item1, item2] = interaction_matrix_cc[item1, item2]
    # print("num of item-item seq edges: ", cnt1)

    #             # adj_matrix_all[num_items + num_users+ word, item] = threshold_weight_edges_iw
    interaction_matrix_cat = np.array(one_hot_encoded_cat)
    for item in range(num_items):
        for cat in range(num_cat_f):
            if interaction_matrix_cat[item, cat] == 1:
                 cnt4+= 1
                # edge_index.append([user, num_users + item])  # User connected to Item
                 edge_index.append([item, (num_items + num_users+ cat)]) 
                 adj_matrix_all[item, num_items + num_users+ cat] = 1
    
    interaction_matrix_level = np.array(one_hot_encoded_level)
    for item in range(num_items):
        for level in range(num_level_f):
            if interaction_matrix_level[item, level] == 1:
                 cnt4 += 1
                # edge_index.append([user, num_users + item])  # User connected to Item
                 edge_index.append([item, (num_items + num_users+ num_cat_f+level)]) 
                 adj_matrix_all[item, num_items + num_users+ num_cat_f+level] = 1
    cnt2= 0
    interaction_matrix_text = np.array(one_hot_encoded_text)
    # kept_words_idx = []
    # for item in range(num_items):
    #     for word in range(num_text_f):
    #         if interaction_matrix_text[item, word] > threshold_weight_edges_iw: # threshold, th = 0.3
    #              cnt2+= 1
    #              cnt4+= 1
    #              kept_words_idx.append(word)
    #             # edge_index.append([user, num_users + item])  # User connected to Item
    #              edge_index.append([item, (num_items + num_users+ num_cat_f + num_level_f+ word)]) 
    #              adj_matrix_all[item, num_items + num_users+ num_cat_f + num_level_f+ word] = threshold_weight_edges_iw
    #             # adj_matrix_all[num_items + num_users+ word, item] = threshold_weight_edges_iw
    # kept_word_node_to_idx = {i: word_idx for i, word_idx in enumerate(kept_words_idx)}
    kept_word_node_to_idx = {}
    for item in range(num_items):
        for word_idx in range(num_text_f):
            if interaction_matrix_text[item, word_idx] >= threshold_weight_edges_iw: # threshold, th = 0.1
            #if interaction_matrix_text[item, word_idx] == 1: # threshold, th = 0.2
                 cnt2+= 1
                #  kept_word_node_to_idx[num_items + num_users+ word_idx] = word_idx  # word node to word idx
                # # edge_index.append([user, num_users + item])  # User connected to Item
                #  edge_index.append([item, (num_items + num_users+ word_idx)]) 
                #  adj_matrix_all[item, num_items + num_users+ word_idx] = interaction_matrix_text[item, word_idx]
                #  #adj_matrix_all[item, num_items + num_users+ word_idx] = 1
                 kept_word_node_to_idx[num_items + num_users+ num_cat_f + num_level_f+ word_idx] = word_idx  # word node to word idx
                # edge_index.append([user, num_users + item])  # User connected to Item
                 edge_index.append([item, (num_items + num_users+ num_cat_f+num_level_f+ word_idx)]) 
                #  adj_matrix_all[item, num_items + num_users+ num_cat_f+num_level_f+ word_idx] = threshold_weight_edges_iw
                 adj_matrix_all[item, num_items + num_users+ num_cat_f+num_level_f+ word_idx] = interaction_matrix_text[item, word_idx]
    # # word word edges
    # cnt3 = 0
    # for word1 in range(num_items, num_items+ num_text_f):
    #     for word2 in range(num_text_f):
    #         if interaction_matrix_text[word1, word2] > threshold_weight_edges_ww and (word1-num_items) != word2: # normalize these values
    #              cnt3 += 1
    #             # edge_index.append([user, num_users + item])  # User connected to Item
    #              edge_index.append([(num_items + num_users+ num_cat_f + num_level_f+ word1-num_items), (num_items + num_users+ num_cat_f + num_level_f+ word2)]) 
    #              adj_matrix_all[num_items + num_users+ num_cat_f + num_level_f+ word1-num_items, num_items + num_users+ num_cat_f + num_level_f+ word2] = threshold_weight_edges_ww
    #             # adj_matrix_all[num_items + num_users+ word2, num_items + num_users+ word1-num_items] = threshold_weight_edges_ww # comment these connections if necessary
    # print("num of item-item edges: ", cnt1) # num of item-item sequential edges 
    print("num of item-word edges: ", cnt2) # num of item-word edges
    # print("num of item-word edges: ", cnt3) # num of word-word edges
    #print("num of edges: ", cnt4)

    #edge_index = torch.tensor(edge_index).t().contiguous()
    # word_cooccurrence = np.zeros((vocab_size, vocab_size))

    adj_sparse = sp.coo_matrix(adj_matrix_all)

    # Step 3: Convert the sparse matrix to a PyTorch Geometric Data object
    edge_index, edge_weight = from_scipy_sparse_matrix(adj_sparse)
    # #Embeddings for users and items
    torch.manual_seed(seed_value)  # Ensure same random initialization
    user_embeddings = torch.nn.Embedding(num_users, embedding_dim)
    item_embeddings = torch.nn.Embedding(num_items, embedding_dim)
    fet_embeddings = torch.nn.Embedding(num_fet, embedding_dim)

    # item_embeddings.weight = torch.nn.Parameter(torch.cat([item_embeddings.weight, item_embeddings_LDA], dim=1))
    # user_embeddings.weight = torch.nn.Parameter(torch.cat([user_embeddings.weight, user_embeddings_LDA], dim=1))
    torch.nn.init.xavier_uniform_(user_embeddings.weight)
    torch.nn.init.xavier_uniform_(item_embeddings.weight)
    torch.nn.init.xavier_uniform_(fet_embeddings.weight)
    # torch.nn.init.normal_(item_embeddings.weight, std=0.1)
    # torch.nn.init.normal_(user_embeddings.weight, std=0.1)
    # x = torch.cat([user_embeddings.weight, item_embeddings.weight], dim=0)
    x = torch.cat([item_embeddings.weight, user_embeddings.weight, fet_embeddings.weight], dim=0)
    # print("shape of x: ", x.shape)  # Output: (N, F)
    # print("shape of edge index: ", edge_index.shape)
    # print(edge_index.max().item())  # Should be less than N
    # print(edge_index.min().item())  # Should be >= 0


    # data = Data(x=x, edge_index=edge_index)
    data = Data(x=x, edge_index=edge_index, edge_attr= edge_weight)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
     # Convert the matrix to a dictionary
    user_interactions = {
        user: {item for item, interacted in enumerate(row) if interacted}
        for user, row in enumerate(interaction_matrix)
    }
    # item_features = {
    #     item: [fet for fet, interacted in enumerate(row) if interacted]
    #     for item, row in enumerate(interaction_matrix_cat)
    # }
    # for item, row in enumerate(interaction_matrix_level):
    #     for level, interacted in enumerate(row):
    #         if interacted:
    #             item_features[item] += [num_cat_f+level]


    # Step 3: Instantiate the model
    # model = LightGCN(num_users, num_items, num_fet, embedding_dim, n_layers, dropout).to(device)
    
    model = GAT(num_users, num_items, num_fet, embedding_dim, n_layers, edge_dropout, node_dropout, heads, seed_value).to(device)
   

    optimizer = optim.Adam(model.parameters(), lr=l_rate)

    # Step 4: Training Loop
    #final_x = data.x
    epochs = n_epochs
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # user_embeddings, item_embeddings, x = model(data.x, data.edge_index) #model(data.x.to(device), data.edge_index.to(device))
        #user_embeddings, item_embeddings, final_x = model(data.x.to(device), data.edge_index.to(device))
        item_embeddings, user_embeddings, fet_embeddings, final_x, final_attention_weights_all_layers, final_edge_index_all_layers = model(edge_index.to(device), edge_weight.to(device)) 
        
        # Dummy loss (you can define a more meaningful loss like BPR, hinge loss, etc.)
        #loss = -torch.mean(user_embeddings @ item_embeddings.t())
         # Sample positive and negative items for BPR Loss
        # user_ids = torch.randint(0, num_users, (num_users,))
        # pos_item_ids = torch.randint(0, num_items, (num_users,))
        # neg_item_ids = torch.randint(0, num_items, (num_users,))
        

        user_ids = torch.randint(0, num_users, (num_users,), device=device)
        pos_item_ids = []
        neg_item_ids = []
        # pos_fet_ids = []
        # neg_fet_ids = []

        for user_id in user_ids:
            # Sample a positive item from the user's actual interactions
            pos_item = random.choice(list(user_interactions[user_id.item()]))  
            pos_item_ids.append(pos_item)
            # pos_fet = random.choice(item_features[pos_item])  
            # pos_fet_ids.append(pos_fet)
            
            # Sample a negative item not in the user's interactions
            neg_item = random.choice([item for item in range(num_items) if item not in list(user_interactions[user_id.item()])])
            neg_item_ids.append(neg_item)
            # negative feature
            # neg_fet = random.choice([fet for fet in range(num_fet) if fet not in item_features[pos_item]])
            #neg_fet = random.choice(item_features[neg_item])  
            # pos_fet_list = []
            # for pos_item2 in list(user_interactions[user_id.item()]):
            #     cf, lf = item_features[pos_item2]
            #     if cf not in pos_fet_list:
            #         pos_fet_list.append(cf)
            #     if lf not in pos_fet_list:
            #         pos_fet_list.append(lf)
            # neg_fet_list = []
            # for fet in range(num_fet):
            #     if fet not in pos_fet_list:
            #         neg_fet_list.append(fet)
            # neg_fet = random.choice(neg_fet_list) 
            # neg_fet_ids.append(neg_fet)

        # Convert lists to tensors
        pos_item_ids = torch.tensor(pos_item_ids, device=device)
        neg_item_ids = torch.tensor(neg_item_ids, device=device)
        # pos_fet_ids = torch.tensor(pos_fet_ids, device=device)
        # neg_fet_ids = torch.tensor(neg_fet_ids, device=device)

        # Gather embeddings for BPR Loss
        u_emb = user_embeddings[user_ids]
        pos_i_emb = item_embeddings[pos_item_ids]
        neg_i_emb = item_embeddings[neg_item_ids]
        # pos_f_emb = fet_embeddings[pos_fet_ids]
        # neg_f_emb = fet_embeddings[neg_fet_ids]

        # Compute BPR Loss
        loss = model.bpr_loss(u_emb, pos_i_emb, neg_i_emb)
        
        loss.backward()
        optimizer.step()
        
        # if epoch % 20 == 0:
        #     print(f"Epoch {epoch}, Loss: {loss.item()}")
        print(f"Epoch {epoch}, Loss: {loss.item()}")

        #data.x = torch.cat([user_embeddings.weight.detach(), item_embeddings.weight.detach()], dim=0)

    # print("from trained model, Item embeddings of 0: ", item_embeddings[0])
    # print("from trained model, user_embeddings of last user: ", user_embeddings[num_users-1])
    base_path = '/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/CDREAM_LGCN/saved_model_GAT_CDREAM_uis_text_GRU_avg_attn_layers_v3_keywords_tfidf/'
    os.makedirs(base_path, exist_ok=True)
    model_filename = f"model_v{version}.pth"
    full_path = os.path.join(base_path, model_filename)
    torch.save(model.state_dict(), full_path)

    #torch.save(model.state_dict(), '/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/CDREAM_LGCN/saved_model_LightGCN_LDA/model.pth')
    # model = GCNRecommender(num_users, num_items, embedding_dim, n_layers)  # Reinitialize the model architecture
    # model.load_state_dict(torch.load('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/GCN/saved_model/model.pth'))
    # torch.save(model, '/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/GCN/saved_model/model_complete.pth')
    # model = GCNRecommender(num_users, num_items)  # Reinitialize the model architecture
    # model.load_state_dict(torch.load('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/GCN/saved_model/model_complete.pth'))


    # #Step 5: Recommendation
    # model.eval()
    # # edge_index = []
    # # k2= 0
    # # num_users_valid = 1
    # # for item in range(num_items):
    # #     edge_index.append([num_users-1, item])  # User connected to Item
    # #     k2 += 1
    # #     if k2==10: break
    # # edge_index = torch.tensor(edge_index).t().contiguous()
    # user_embeddings, item_embeddings, x2 = model(data.x.to(device), data.edge_index.to(device))
    # # print(item_embeddings[0])
    # # print("Item embedding of 0: ", item_embeddings[0])
    # # print("user_embedding of last user: ", user_embeddings[num_users-1])
    # recommended_items = model.recommend(user_embeddings, item_embeddings, top_k=15)

    # print("Recommended items for each user:")
    # for user_id, items in enumerate(recommended_items):
    #     list2 = []
    #     for item1 in items: 
    #         list2.append(item_dict_one_hot[item1.item()]) 
    #     print(f"User: {user_dict_one_hot[user_id]}, Item IDs: {list2}")
    print(final_x.shape)  
    print("num of nodes: ", final_x.shape[0])
    print("num of edges: ", cnt4)
    print("len of attention weights: ", len(final_attention_weights_all_layers))
    print("num of edges in attention weights: ", len(final_attention_weights_all_layers[0]))
    print("shape of edge index out: ", final_edge_index_all_layers[0].shape)
    return model, data, final_x, final_attention_weights_all_layers, final_edge_index_all_layers, kept_word_node_to_idx

if __name__ == '__main__':
    train_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/train_data_all_CR.json', orient='records', lines= True)
    start = time.time()
    print("Start")
    dataTrain, dataTest, dataTotal, item_list, item_dict, reversed_item_dict, one_hot_encoded_train, one_hot_encoded_df_train, item_dict_one_hot, reversed_item_dict_one_hot, user_dict_one_hot, one_hot_encoded_train2, user_dict_one_hot_train, reversed_user_dict_one_hot_train = preprocess_data(train_data)
    # df = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/CDREAM_LGCN/processed_courses_v2.json', orient='records', lines= True)
    # emb_dim_LDA = 128
    # max_df1 = 0.85
    # min_df1 = 1
    # random_state = 0
    # num_users_train = one_hot_encoded_train.shape[0]
    num_items = one_hot_encoded_train.shape[1]
    # doc_topic_matrix, doc_topic_df, item_embeddings, model, item_dict_idx_to_cid, item_dict_cid_to_idx = CDREAM_LGCN(df, emb_dim_LDA, max_df1, min_df1, random_state)
    # user_embeddings_LDA, item_embeddings_LDA = create_embedding_for_training_users(one_hot_encoded_train, num_users_train, item_embeddings, item_dict_idx_to_cid, item_dict_cid_to_idx, reversed_item_dict_one_hot, num_items)
    # user_embeddings_LDA2 = torch.tensor(user_embeddings_LDA, dtype=torch.float32)
    # item_embeddings_LDA2 = torch.tensor(item_embeddings_LDA, dtype=torch.float32)
    # df = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/CDREAM_LGCN/processed_courses_v3.json', orient='records', lines=True)
    df = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/CDREAM_LGCN/course_info_desc_keywords.json', orient='records', lines=True)
    #course_concept_path = "/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/CDREAM_LGCN/course_concepts_v2.csv"
    one_hot_encoded_cat, one_hot_encoded_level, cat_dict_one_hot, level_dict_one_hot, reversed_dict_cat_to_idx, reversed_dict_level_to_idx, one_hot_df_cat, one_hot_df_level = convert_side_info_to_one_hot_encoding(df, reversed_item_dict_one_hot, num_items)
    # print(one_hot_encoded_cat.shape)
    # print(one_hot_encoded_level.shape)
    # print(reversed_dict_cat_to_idx.keys())
    # print(one_hot_df_cat)
    # print(one_hot_encoded_cat)
    window_s = 10
    adj_mat, vocab_size, vocab_dict_idx_to_wrd, vocab_dict_wrd_to_idx = get_adj_matrix_text_v3(df, reversed_item_dict_one_hot, window_s)
    #print(adj_mat)
    #print(adj_mat.shape)
    print(vocab_size)
    # dense_adj_matrix_text = adj_mat.todense()
    # #print(dense_adj_matrix_text)
    # print(dense_adj_matrix_text.shape)  # (num_items + vocab, vocab)
    # dense_adj_matrix_text2= np.array(dense_adj_matrix_text)
    dense_adj_matrix_text = adj_mat.toarray()
    adj_matrix_text = normalize(dense_adj_matrix_text, norm='l2')
    #print(adj_matrix_text)
    threshold_weight_edges_iw = 0.1
    threshold_weight_edges_ww = 0.1
    # cc_seq_matrix = measure_sequence_of_courses(dataTotal, reversed_item_dict_one_hot)  # course to id
    # print(cc_seq_matrix.shape)
    # threshold_weight_edges_cc= 0.2

    n_layers = 3
    embedding_dim = 128
    epochs = 200
    lr = 0.01
    edge_dropout = 0
    node_dropout = 0
    n_heads = 2
    version = 1
    seed_value = 42
    
    model, data, final_x, final_attention_weights_all_layers, final_edge_index_all_layers, kept_word_node_to_idx = train_model(one_hot_encoded_train, one_hot_encoded_cat, one_hot_encoded_level, adj_matrix_text, n_layers, embedding_dim, epochs, lr, edge_dropout, node_dropout, n_heads, threshold_weight_edges_iw,  threshold_weight_edges_ww, seed_value, version)
    end = time.time()
    print("time: ", end-start)