import torch

from models import SpKBGATModified, SpKBGATConvOnly
from models import SpKBGATModified_modified
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy

from preprocess import read_entity_from_id, read_relation_from_id, init_embeddings, build_data,  build_data_modified
from create_batch import Corpus
from utils import save_model

import random
import argparse
import os
import sys
import logging
import time
import pickle
import ogb
from ogb.linkproppred import LinkPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from torch_geometric.data import NeighborSampler
import torch_geometric.transforms as T
import pdb

# %%
# %%from torchviz import make_dot, make_dot_from_trace


def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="./data/WN18RR", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3600, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=False)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/ogb12/", help="Folder name to save the models.")
    args.add_argument("-outprint_file", "--outprint_file",
                      default="./outprint12.txt", help="File name to print information.")
    args.add_argument("-path_checkpoint", "--path_checkpoint",
                      default="./checkpoints/ogb12/conv/trained_25.pth", help="last epoch")
    args.add_argument("-start_epoch", "--start_epoch",
                      default=26, type=int, help="start epoch num")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=262144, help="Batch size for GAT")  # 2^18
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropouti probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=5, help="Margin used in hinge loss")
    args.add_argument("-layer_num", "--layer_num", type=int,
                      default= 4, help="Number of layers of a GNN")
    args.add_argument("-device", "--GPU_device", type=int,
                      default=5, help="the primary GPU for training")
    args.add_argument('-multi_gpu', "--multi_gpu", type = bool,default = False)
    args.add_argument('-sampled', "--sampled", type=int, default=2)

    # arguments for convolution networki
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=8192, help="Batch size for conv") #2^13
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layer")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=500,
                      help="Number of output channels in conv layer")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.0, help="Dropout probability for convolution layer")
    args = args.parse_args()
    return args


args = parse_args()
# %%

print('args.get_2hop:', args.get_2hop)
print('args.use_2hop:', args.use_2hop)
print('args.pretrained_emb:', args.pretrained_emb)
print('args.nheads_GAT:', args.nheads_GAT[0])
print('args.layer_num:',args.layer_num)
print('args.GPU_device:', args.GPU_device)
print('args.multi_gpu:', args.multi_gpu)
print('args.outprint_file:', args.outprint_file)
print('args.output_folder:', args.output_folder)
print('args.sampled:', args.sampled)
print('args.path_checkpoint:', args.path_checkpoint)
print('args.start_epoch:', args.start_epoch)


# with open('./outprint.txt','w') as f:
#     f.write('args.nheads_GAT:{}\n'.format(args.nheads_GAT))
#     f.write('args.layer_num:{}\n'.format(args.layer_num))
#     f.write('args.multi_gpu:{}\n'.format(args.multi_gpu))

device = torch.device('cuda:{}'.format(args.GPU_device))

def load_data(args, split_dict, relation_array, entity_array):
    # train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train = build_data(
    #     args.data, is_unweigted=False, directed=True)
    train_data, validation_data, test_data, headTailSelector, unique_entities_train = build_data_modified(
         split_dict, relation_array, is_unweigted=False, directed=True)

    if args.pretrained_emb:
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, 'entity2vec.txt'),
                                                                 os.path.join(args.data, 'relation2vec.txt'))
        print("Initialised relations and entities from TransE")
        #f.write("Initialised relations and entities from TransE\n")
    else:
        # entity_embeddings = np.random.randn(
        #     len(entity2id), args.embedding_size)
        # relation_embeddings = np.random.randn(
        #     len(relation2id), args.embedding_size)
        entity_embeddings = np.random.randn(
            nentity, args.embedding_size)
        relation_embeddings = np.random.randn(
            nrelation, args.embedding_size)
        print("Initialised relations and entities randomly")
        #f.write("Initialised relations and entities randomly\n")
    # corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector,
    #                 args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop)

    corpus = Corpus(args, train_data, validation_data, test_data, entity_array, relation_array, headTailSelector,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop)

    return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings)


dataset = LinkPropPredDataset(name = 'ogbl-wikikg2')
split_dict = dataset.get_edge_split()
# split_dict['train']['head']  = split_dict['train']['head'][:500000]   #从 16million+中的training triples 取前面 50w个
# split_dict['train']['relation']  = split_dict['train']['relation'][:500000]
# split_dict['train']['tail']  = split_dict['train']['tail'][:500000]
#
entity_array = np.union1d(split_dict['train']['head'],split_dict['train']['tail'])  # 2,500,604 entities
relation_array = np.array(list(set(split_dict['train']['relation'])))  # 535 relation types
nentity = dataset.graph['num_nodes']  # 2,500,604 nodes
nrelation = int(max(dataset.graph['edge_reltype'])[0])+1    # 535 relation types

# with open('./outprint.txt','a') as f:
Corpus_, entity_embeddings, relation_embeddings = load_data(args,split_dict, relation_array, entity_array)  #时间久


if(args.get_2hop):
    file = args.data + "/2hop.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(Corpus_.node_neighbors_2hop, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)     # {source:{distance:[((relations,),(entities, )),....]} }


if(args.use_2hop):
    print("Opening node_neighbors pickle object")
    file = args.data + "/2hop.pickle"
    with open(file, 'rb') as handle:
        node_neighbors_2hop = pickle.load(handle)

entity_embeddings_copied = deepcopy(entity_embeddings)
relation_embeddings_copied = deepcopy(relation_embeddings)

print("Initial entity dimensions {} , relation dimensions {}".format(
    entity_embeddings.size(), relation_embeddings.size()))

# with open('./outprint.txt','a') as f:
#     f.write("Initial entity dimensions {} , relation dimensions {}\n".format(
#     entity_embeddings.size(), relation_embeddings.size()))

# %%

CUDA = torch.cuda.is_available()


def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed, device):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=1)

    y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).to(device)

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


# def train_gat(args):
#
#     # Creating the gat model here.
#     ####################################
#
#     print("Defining model")
#     f.write('Defining model\n')
#
#     print("\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
#     f.write("\nModel type -> GAT layer with {} heads used , Initital Embeddings training\n".format(args.nheads_GAT[0]))
#
#     # model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
#     #                             args.drop_GAT, args.alpha, args.nheads_GAT)
#
#     model_gat_modified = SpKBGATModified_modified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
#                                 args.drop_GAT, args.alpha, args.nheads_GAT, args.layer_num)
#     if CUDA:
#         model_gat_modified.to(device)
#
#     optimizer = torch.optim.Adam(
#         model_gat_modified.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)
#
#     scheduler = torch.optim.lr_scheduler.StepLR(
#         optimizer, step_size=500, gamma=0.5, last_epoch=-1)
#
#     gat_loss_func = nn.MarginRankingLoss(margin=args.margin)
#
#     # current_batch_2hop_indices = torch.tensor([])
#     # if(args.use_2hop):
#     #     current_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args, \
#     #                                                                       Corpus_.unique_entities_train, node_neighbors_2hop)
#                                                         # tensor[[source_entity_train, relation_source, relation_target, target_entity_train],[], ...]
#     # if CUDA:
#     #     current_batch_2hop_indices = Variable(
#     #         torch.LongTensor(current_batch_2hop_indices)).cuda()
#     # else:
#     #     current_batch_2hop_indices = Variable(
#     #         torch.LongTensor(current_batch_2hop_indices))
#
#     epoch_losses = []   # losses of all epochs
#     print("Number of epochs {}".format(args.epochs_gat))
#     f.write("Number of epochs {}\n".format(args.epochs_gat))
#
#     for epoch in range(args.epochs_gat):
#
#         print("\nepoch-> ", epoch)               # shuffle for every epoch
#         f.write("\nepoch-> {}\n".format(epoch))
#
#         random.shuffle(Corpus_.train_triples)   # train_triples: [(e1_id,r_id,e2_id),(), ...]
#         Corpus_.train_indices = np.array(
#             list(Corpus_.train_triples)).astype(np.int32)
#
#         # model_gat.train()  # getting in training mode
#         model_gat_modified.train()  # getting in training mode
#         start_time = time.time()
#         epoch_loss = []
#
#         if len(Corpus_.train_indices) % args.batch_size_gat == 0:
#             num_iters_per_epoch = len(
#                 Corpus_.train_indices) // args.batch_size_gat
#         else:
#             num_iters_per_epoch = (
#                 len(Corpus_.train_indices) // args.batch_size_gat) + 1
#
#         for iters in range(num_iters_per_epoch):
#             start_time_iter = time.time()
#             train_indices, train_values = Corpus_.get_iteration_batch(iters)    # simple sampling methods: [ : valid triples for one batch, :invalid triples for replacing head entity,  : invalid for replacing tail entity]
#
#             if CUDA:
#                 train_indices = Variable(
#                     torch.LongTensor(train_indices)).to(device)
#                 train_values = Variable(torch.FloatTensor(train_values)).to(device)
#
#             else:
#                 train_indices = Variable(torch.LongTensor(train_indices))
#                 train_values = Variable(torch.FloatTensor(train_values))
#
#             start_time_forward = time.time()
#             # forward pass
#             # entity_embed, relation_embed = model_gat(
#             #     Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices)   #时间
#             len_pos_triples = int(
#                 train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))
#             batch_inputs = train_indices[:len_pos_triples]
#
#             # unique_entities_number = Corpus_.get_unique_entites_number(batch_inputs)
#             if torch.cuda.device_count() > 1 and args.multi_gpu:
#                 model_gat_modified = nn.DataParallel(model_gat_modified, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
#             # model_gat_modified.to(parallel_device)
#             entity_embed, relation_embed = model_gat_modified(Corpus_, batch_inputs, args.multi_gpu, device)  # 时间
#
#             print("Iteration-> {0}  , forward_time-> {1:.4f}".format(
#                 iters, time.time() - start_time_forward))
#             f.write("Iteration-> {0}  , forward_time-> {1:.4f}\n".format(
#                 iters, time.time() - start_time_forward))
#
#             optimizer.zero_grad()
#
#             # loss = batch_gat_loss(
#             #     gat_loss_func, train_indices, entity_embed, relation_embed)
#             loss = batch_gat_loss(
#                 gat_loss_func, train_indices, entity_embed, relation_embed, device)
#
#             start_time_backward = time.time()
#             loss.backward()
#             print("Iteration-> {0}  , backward_time-> {1:.4f}".format(
#                 iters, time.time() - start_time_backward))
#             f.write("Iteration-> {0}  , backward_time-> {1:.4f} \n".format(
#                 iters, time.time() - start_time_backward))
#
#             optimizer.step()
#
#             epoch_loss.append(loss.data.item())
#
#             end_time_iter = time.time()
#
#             print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f} \n".format(
#                 iters, end_time_iter - start_time_iter, loss.data.item()))
#             f.write("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}\n".format(
#                 iters, end_time_iter - start_time_iter, loss.data.item()))
#
#         scheduler.step()
#         print("Epoch {} , average loss {} , epoch_time {}".format(
#             epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
#         f.write("Epoch {} , average loss {} , epoch_time {}\n".format(
#             epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
#
#         epoch_losses.append(sum(epoch_loss) / len(epoch_loss))
#
#         save_model(model_gat_modified, args.data, epoch,
#                    args.output_folder)


def train_conv(args):


    # Creating convolution model here.
    ####################################


    print("Defining model")
    f.write('Defining model\n')
    model_gat = SpKBGATModified_modified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT, args.layer_num)
    print("Only Conv model trained")
    f.write('Only Conv model trained\n')
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)

    if CUDA:
        model_conv.to(device)
        model_gat.to(device)


    # model_gat.load_state_dict(torch.load(
    #     '{}/trained_{}.pth'.format(args.output_folder, args.epochs_gat - 1)), strict=False)
    model_conv.load_state_dict(torch.load(args.path_checkpoint), strict=False)
    # model_conv.final_entity_embeddings = model_gat.final_entity_embeddings
    # model_conv.final_relation_embeddings = model_gat.final_relation_embeddings


    Corpus_.batch_size = args.batch_size_conv
    Corpus_.invalid_valid_ratio = int(args.valid_invalid_ratio_conv)


    optimizer = torch.optim.Adam(
        model_conv.parameters(), lr=args.lr, weight_decay=args.weight_decay_conv)
    #optimizer.load_state_dict(torch.load(args.path_checkpoint)['optimizer'], strict=False)


    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.5, last_epoch=-1)


    margin_loss = torch.nn.SoftMarginLoss()


    epoch_losses = []   # losses of remanent epochs
    print("Number of residual epochs {}".format(args.epochs_conv-args.start_epoch))
    f.write('Number of residual epochs {}\n'.format(args.epochs_conv-args.start_epoch))

    for epoch in range(args.start_epoch, args.epochs_conv):
        print("\nepoch-> ", epoch)
        f.write('epoch-> {}\n'.format(epoch))
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)


        model_conv.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []


        if len(Corpus_.train_indices) % args.batch_size_conv == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_conv
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_size_conv) + 1


        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if CUDA:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).to(device)
                train_values = Variable(torch.FloatTensor(train_values)).to(device)

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            preds = model_conv(
                Corpus_, Corpus_.train_adj_matrix, train_indices)

            optimizer.zero_grad()

            loss = margin_loss(preds.view(-1), train_values.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            #print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}\n".format(
            #    iters, end_time_iter - start_time_iter, loss.data.item()))
            f.write("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}\n".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}\n".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        f.write("Epoch {} , average loss {} , epoch_time {}\n".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))

        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        save_model(model_conv, args.data, epoch,
                   args.output_folder + "conv/")



def evaluate_conv(args, unique_entities):
    model_conv = SpKBGATConvOnly(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                 args.drop_GAT, args.drop_conv, args.alpha, args.alpha_conv,
                                 args.nheads_GAT, args.out_channels)
    model_conv.load_state_dict(torch.load(
        '{0}conv/trained_{1}.pth'.format(args.output_folder, args.epochs_conv - 1)), strict=False)

    model_conv.to(device)
    model_conv.eval()
    with torch.no_grad():
        # Corpus_.get_validation_pred(args, model_conv, unique_entities)
        Corpus_.get_validation_pred_modified(args, model_conv, unique_entities,f)





with open(args.outprint_file,'a') as f:
#     t1= time.time()
#     train_gat(args)
#     t2 = time.time()
#     f.write("GAT training time: {}\n". format(t2-t1))
#
    t1= time.time()
    train_conv(args)
    t2 = time.time()
    f.write("ConvKB training time: {}\n". format(t2-t1))

    t1= time.time()
    evaluate_conv(args, Corpus_.unique_entities_train)
    t2 = time.time()
    f.write("Evaluation time: {}\n". format(t2-t1))
