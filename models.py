import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import SpGraphAttentionLayer, ConvKB
from layers import SpGraphAttentionLayer_modified
import numpy as np
import utils
import pdb

CUDA = torch.cuda.is_available()  # checking cuda availability


class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nheads * nhid, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop):
        x = entity_embeddings

        edge_embed_nhop = relation_embed[    #edge_type_nhop [[source_relation1, target_relation2], [], ..]
            edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]   # relation_source, relation_target

        x = torch.cat([att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)
                       for att in self.attentions], dim=1)
        x = self.dropout_layer(x)

        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]
        edge_embed_nhop = out_relation_1[
            edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]]  # list相加: dim=1

        x = F.elu(self.out_att(x, edge_list, edge_embed,
                               edge_list_nhop, edge_embed_nhop))
        return x, out_relation_1

class SpGAT_modified(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads, layer_num):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention
            layer_num -> number of layers (default >= 2)

        """
        super(SpGAT_modified, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_num = layer_num
        self.nheads = nheads
        # self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
        #                                          nhid,
        #                                          relation_dim,
        #                                          dropout=dropout,
        #                                          alpha=alpha,
        #                                          concat=True)
        #                    for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

        self.attentions_modified = []
        for l in range(self.layer_num):   # initialize all attention layers with multiple heads
            if l == 0:   # the first layer
                for _ in range(nheads):
                    self.attentions_modified.append(SpGraphAttentionLayer_modified(num_nodes, nfeat,
                                                         nhid,
                                                         relation_dim,
                                                         dropout=dropout,
                                                         alpha=alpha,
                                                         concat=True))
            elif l == layer_num-1:   # the final layer
                self.attentions_modified.append(SpGraphAttentionLayer_modified(num_nodes, nhid * nheads,
                                                                       nfeat,
                                                                       nhid,
                                                                       dropout=dropout,
                                                                       alpha=alpha,
                                                                       concat=False))
            else:
                for _ in range(nheads):
                    self.attentions_modified.append(SpGraphAttentionLayer_modified(num_nodes,  nhid * nheads,
                                                                           nhid,
                                                                           nhid,
                                                                           dropout=dropout,
                                                                           alpha=alpha,
                                                                           concat=True))

        for i, attention in enumerate(self.attentions_modified):
            self.add_module('attention_{}'.format(i), attention)


        # W matrix to convert h_input to h_output dimension
        # self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))

         # if only 2 layers -> one value, elif more than 2 layers -> two values
        self.W1 = nn.Parameter(torch.zeros(size=(relation_dim,  nhid)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        if self.layer_num > 2:
            self.W2 = nn.Parameter(torch.zeros(size=(nhid, nhid)))
            nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.W3 = nn.Parameter(torch.zeros(size=(nhid, nfeat)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)

        self.W_source = nn.Parameter(torch.zeros(size=(nfeat,  nhid * nheads)))
        nn.init.xavier_uniform_(self.W_source.data, gain=1.414)

        self.W_target = nn.Parameter(torch.zeros(size=(nfeat, nfeat)))
        nn.init.xavier_uniform_(self.W_target.data, gain=1.414)

        # nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
        #                                      nheads * nhid, nheads * nhid,
        #                                      dropout=dropout,
        #                                      alpha=alpha,
        #                                      concat=False
        #                                      )


    def forward(self, Corpus_, entity_embeddings, relation_embed, edge_list, edge_type, multi_gpu, device):
        x = entity_embeddings

        # edge_type_nhop [[source_relation1, target_relation2], [], ..]
        # edge_embed_nhop = relation_embed[
        #     edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]   # relation_source, relation_target

        # Self-attention on the nodes - Shared attention mechanism
        # edge = torch.cat((edge_list[:, :], edge_list_nhop[:, :]), dim=1)  # size : 2 × N   [[rows], [columns]]
        # edge_embed = torch.cat(
        #     (edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)  # size: N × dim_relation

        # edge_embed_nhop = relation_embed[  # edge_type_nhop [[source_relation1, target_relation2], [], ..]
        #                       edge_type_nhop[:, 0]] + relation_embed[
        #                       edge_type_nhop[:, 1]]  # relation_source, relation_target
        # x = torch.cat([att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)
        #                for att in self.attentions], dim=1)

        # x = self.dropout_layer(x)
        #
        # out_relation_1 = relation_embed.mm(self.W)
        #
        # edge_embed = out_relation_1[edge_type]
        # edge_embed_nhop = out_relation_1[
        #                       edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]]  # list相加: dim=1
        #
        # x = F.elu(self.out_att(x, edge_list, edge_embed,
        #                        edge_list_nhop, edge_embed_nhop))


        # edge_list:  [[rows], [columns]],  size: (2 , N)


        #tansfer scatter entity id in the mini-batch to continous entity id

        start_time1 = time.time()
        combined = torch.cat((edge_list[0],edge_list[1]))
        scatter_entity, counts = combined.unique(return_counts=True)  # automatic sorting
        print("proprocess0_1 for forward propogation of GNN-> {:.4f} ".format(time.time() - start_time1))
        
        start_time2 = time.time()
        scatter_target = torch.unique(edge_list[0])  # automatic sorting
        scatter_source = torch.tensor(np.setdiff1d(edge_list[1].to('cpu'),edge_list[0].to('cpu'))).to(device) # utomatic sorting, on gpu
        #scatter_source = torch.unique(torch.tensor([s for s in edge_list[1] if s not in scatter_target])) # automatic sorting, on cpu
        print("proprocess0_2 for forward propogation of GNN-> {:.4f} ".format(time.time() - start_time2))

        start_time = time.time()
        scatter_continuous_dict = utils.scatter_map_continuous(scatter_entity)   # key: scatter_id, value:continuous_id
        entity_continuous_list = []
        entity_target_continuous = []
        entity_source_continuous = []
        edge_list = edge_list.to('cpu')
        for i in range(len(edge_list[0])):
            entity_target_continuous.append(scatter_continuous_dict[edge_list[0][i].item()])
        entity_continuous_list.append(entity_target_continuous)
        for i in range(len(edge_list[1])):
            entity_source_continuous.append(scatter_continuous_dict[edge_list[1][i].item()])
        entity_continuous_list.append(entity_source_continuous)
        if (CUDA):
            entity_continuous_list = torch.tensor(entity_continuous_list).to(device)
        print("proprocess1 for forward propogation of GNN-> {:.4f} ".format(time.time() - start_time))

        start_time = time.time()
        if len(scatter_source) != 0:
            entity_source_embed = entity_embeddings[scatter_source, :].mm(self.W_source)
        entity_target_embed = entity_embeddings[scatter_target, :].mm(self.W_target)

        edge_embed = relation_embed[edge_type]  # edge_embed (N,dim_relation)
        edge_h = torch.cat(  # edge_h: (2*in_dim + nrela_dim) x E
            (x[edge_list[0, :], :], x[edge_list[1, :], :], edge_embed[:, :]), dim=1).t()
        print("proprocess2 for forward propogation of GNN-> {:.4f} ".format(time.time() - start_time))

        start_time = time.time()
        scatter_source = scatter_source.to('cpu')
        scatter_target = scatter_target.to('cpu')
        for l in range(self.layer_num):
            if l == self.layer_num-1:      # the final layer
                x = self.attentions_modified[-1](scatter_entity, entity_continuous_list, edge_h,  multi_gpu)
                out_relation_1 = out_relation_1.mm(self.W3)
                x = F.elu(x)
            else:
                x = torch.cat([att(scatter_entity, entity_continuous_list, edge_h, multi_gpu)
                                for att in self.attentions_modified[l * self.nheads: (l + 1) * self.nheads]], dim=1)  # update entity_embeddings
                x = self.dropout_layer(x)
                if len(scatter_source) != 0:
                    for i in range(len(scatter_source)):
                        x[scatter_continuous_dict[scatter_source[i].item()],:] = entity_source_embed[i]  # reassign the unique source entities
                    
                if l == 0:         # the first layer
                    out_relation_1 = relation_embed.mm(self.W1)   # update relation embeddings
                else:
                    out_relation_1 = out_relation_1.mm(self.W2)
                edge_embed = out_relation_1[edge_type]
                edge_h = torch.cat(
                    (x[entity_continuous_list[0, :],:], x[entity_continuous_list[1, :], :], edge_embed[:, :]), dim=1).t()
        print("forward propogation of GNN-> {:.4f} ".format( time.time() - start_time))

        start_time = time.time()
        for i in range(len(scatter_target)):  # target entity
            target_embed = x[scatter_continuous_dict[scatter_target[i].item()], :] \
                           + entity_target_embed[i]
           # entity_embeddings[scatter_target[i]].data = F.normalize(target_embed.unsqueeze(0), p=2, dim=1)[0].data
            entity_embeddings[scatter_target[i]].data = target_embed.data
        print("push updated embedding of target entity tp the history embeddings -> {:.4f} ".format(time.time() - start_time))
        return entity_embeddings, out_relation_1   #nfeat, nfeat


class SpKBGATModified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.alpha = alpha      # For leaky relu

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)

        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop):    # train_indices_nhop: [[source_entity_train, relation_source, relation_target, target_entity_train],[], ...]
        # getting edge list                                               # batch_inputs:  [ : valid triples for one batch, :invalid triples for replacing head entity,  : invalid for replacing tail entity]
        edge_list = adj[0]   # [[e2_id,..],[e1_id,..]]
        edge_type = adj[1]   # [r_id, ..]

        edge_list_nhop = torch.cat(       #unsqueeze(-1): add a new dimension after the last index
            (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
        edge_type_nhop = torch.cat(
            [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)

        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()

        edge_embed = self.relation_embeddings[edge_type]

        start = time.time()

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()  #detach a tensor from the current computational graph when we don't require a gradient.

        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1)

        out_entity_1, out_relation_1 = self.sparse_gat_1(
            Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings,
            edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop)

        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()  # unique value set
        mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        mask[mask_indices] = 1.0

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity_1.data
        self.final_relation_embeddings.data = out_relation_1.data

        return out_entity_1, out_relation_1


class SpKBGATModified_modified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT, layer_num):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.alpha = alpha      # For leaky relu

        # self.final_entity_embeddings = nn.Parameter(
        #     torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        # self.final_relation_embeddings = nn.Parameter(
        #    torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))
        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_in_dim))
        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_in_dim))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)  # Size([2500604, 50])
        self.relation_embeddings = nn.Parameter(initial_relation_emb)  #Size([535, 50])
        # self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
        #                           self.drop_GAT, self.alpha, self.nheads_GAT_1)

        self.sparse_gat_1_modified = SpGAT_modified(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1, layer_num)

        # self.W_entities = nn.Parameter(torch.zeros(
        #         size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))

    def forward(self, Corpus_, batch_inputs, multi_gpu, device):    # train_indices_nhop: [[source_entity_train, relation_source, relation_target, target_entity_train],[], ...]
        # getting edge list                                               # batch_inputs:  [ : valid triples for one batch]
        # edge_list = adj[0]   # [[e2_id,..],[e1_id,..]]
        # edge_type = adj[1]   # [r_id, ..]
        #
        # edge_list_nhop = torch.cat(       #unsqueeze(-1): add a new dimension after the last index
        #     (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
        # edge_type_nhop = torch.cat(
        #     [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)

        edge_list = torch.concat((batch_inputs[:,2].unsqueeze(-1),batch_inputs[:,0].unsqueeze(-1)),dim=1).t()  # [[e2_id,..],[e1_id,..]], size(2,272115)
        edge_type = batch_inputs[:,1]  # [r_id, ..] , size(272115)

        #if(CUDA):
            #edge_list = edge_list.cuda()
            #edge_type = edge_type.cuda()
            # edge_list_nhop = edge_list_nhop.cuda()
            # edge_type_nhop = edge_type_nhop.cuda()

        # edge_embed = self.relation_embeddings[edge_type]

        start = time.time()

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()  #detach a tensor from the current computational graph when we don't require a gradient.

        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1)

        start_time = time.time()
        out_entity_1, out_relation_1 = self.sparse_gat_1_modified(
            Corpus_,  self.entity_embeddings, self.relation_embeddings,
            edge_list, edge_type, multi_gpu, device)
        print("SpGAT_modified_time-> {:.4f} ".format( time.time() - start_time))

        # mask_indices = torch.unique(batch_inputs[:, 2]).cuda()  # the set containing unique target entities (updated within this batch_inputs)
        # mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        # mask[mask_indices] = 1.0
        #
        # entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        # out_entity_1 = entities_upgraded + \
        #     mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1
        #
        # out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity_1.data      # torch.Size([2500604, 200])
        self.final_relation_embeddings.data = out_relation_1.data   # torch.Size([535, 200])

        return out_entity_1, out_relation_1


class SpKBGATConvOnly(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.drop_conv = drop_conv
        self.alpha = alpha      # For leaky relu
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels

        #self.final_entity_embeddings = nn.Parameter(
        #    torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))
        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_in_dim))
        #self.final_relation_embeddings = nn.Parameter(
        #    torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))
        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_in_dim))

        #self.convKB = ConvKB(self.entity_out_dim_1 * self.nheads_GAT_1, 3, 1,
        #                     self.conv_out_channels, self.drop_conv, self.alpha_conv)
        self.convKB = ConvKB(self.entity_in_dim, 3, 1,
                             self.conv_out_channels, self.drop_conv, self.alpha_conv)

    def forward(self, Corpus_, adj, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv

    def batch_test(self, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv
