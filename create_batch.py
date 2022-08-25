import torch
import numpy as np
from collections import defaultdict
import time
import queue
import random
import pdb


class Corpus:
    def __init__(self, args, train_data, validation_data, test_data, entity_array,
                 relation_array, headTailSelector, batch_size, valid_to_invalid_samples_ratio, unique_entities_train, get_2hop=False):
        self.entity_array = entity_array
        self.relation_array = relation_array

        self.train_triples = train_data[0]

        # Converting to sparse tensor
        adj_indices = torch.LongTensor(
            [train_data[1][0], train_data[1][1]])  # [[rows],[columns]]
        adj_values = torch.LongTensor(train_data[1][2])  # [relations]
        self.train_adj_matrix = (adj_indices, adj_values)

        # adjacency matrix is needed for train_data only, as GAT is trained for training data
        # self.validation_triples = validation_data[0]
        # self.test_triples = test_data[0]
        self.validation_triples = [(validation_data[0][i][0],validation_data[0][i][2],validation_data[0][i][3]) for i in range(len(validation_data[0]))]
        self.test_triples = [(test_data[0][i][0],test_data[0][i][2],test_data[0][i][3]) for i in range(len(test_data[0]))]
        self.test_data = test_data
        self.validation_data = validation_data

        self.headTailSelector = headTailSelector  # for selecting random entities
        # self.entity2id = entity2id
        # self.id2entity = {v: k for k, v in self.entity2id.items()}
        # self.relation2id = relation2id
        # self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.batch_size = batch_size
        # ratio of valid to invalid samples per batch for training ConvKB Model
        self.invalid_valid_ratio = int(valid_to_invalid_samples_ratio)

        if(get_2hop):
            self.graph = self.get_graph()  # for train data
            self.node_neighbors_2hop = self.get_further_neighbors()   #时间久 , {source:{distance:[((relations,),(entities, )),....]} }

        # self.unique_entities_train = [self.entity2id[i]
        #                               for i in unique_entities_train]
        self.unique_entities_train = unique_entities_train

        self.train_indices = np.array(
            list(self.train_triples)).astype(np.int32)
        # These are valid triples, hence all have value 1
        self.train_values = np.array(
            [[1]] * len(self.train_triples)).astype(np.float32)

        self.validation_indices = np.array(
            list(self.validation_triples)).astype(np.int32)
        self.validation_values = np.array(
            [[1]] * len(self.validation_triples)).astype(np.float32)

        self.test_indices = np.array(list(self.test_triples)).astype(np.int32)
        self.test_values = np.array(
            [[1]] * len(self.test_triples)).astype(np.float32)

        self.valid_triples_dict = {j: i for i, j in enumerate(
            self.train_triples + self.validation_triples + self.test_triples)}
        print("Total triples count {}, training triples {}, validation_triples {}, test_triples {}".format(len(self.valid_triples_dict), len(self.train_indices),
                                                                                                           len(self.validation_indices), len(self.test_indices)))
    
        # For training purpose
        self.batch_indices = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
        self.batch_values = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)
        
        #self.get_graph_degree()
        # For Sampling
        if args.sampled != 0:
            self.graph ,self.source_out, self.target_out = self.get_graph_modified()  # for train data

        if args.sampled == 1:
            self.unsampled_graph = self.graph
            self.unsampled_source = list(self.unsampled_graph.keys())  # source entities of unsampled triples

    def get_iteration_batch(self, iter_num):
        if (iter_num + 1) * self.batch_size <= len(self.train_indices):  #not the last iteration
            self.batch_indices = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            self.batch_size * (iter_num + 1))

            self.batch_indices[:self.batch_size,
                               :] = self.train_indices[indices, :]  # train_indices-> train_triples : [(e1_id,r_id,e2_id),(), ...]
            self.batch_values[:self.batch_size,
                              :] = self.train_values[indices, :]

            last_index = self.batch_size

            if self.invalid_valid_ratio > 0:
                # random_entities = np.random.randint(
                #     0, self.nentity, last_index * self.invalid_valid_ratio)
                random_entities = np.random.choice(
                      self.entity_array, last_index * self.invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            # random_entities[current_index] = np.random.randint(
                            #     0, self.nentity)
                            random_entities[current_index] = np.random.choice(self.entity_array,1)
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * \
                            (self.invalid_valid_ratio // 2) + \
                            (i * (self.invalid_valid_ratio // 2) + j)

                        while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            # random_entities[current_index] = np.random.randint(
                            #     0, self.nentity)
                            random_entities[current_index] = np.random.choice(self.entity_array, 1)
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                return self.batch_indices, self.batch_values

            return self.batch_indices, self.batch_values    # [ : valid triples for one batch, :invalid triples for replacing head entity,  : invalid for replacing tail entity]

        else:    # the last/only one  iteration
            last_iter_size = len(self.train_indices) - \
                self.batch_size * iter_num
            self.batch_indices = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            len(self.train_indices))
            self.batch_indices[:last_iter_size,
                               :] = self.train_indices[indices, :]
            self.batch_values[:last_iter_size,
                              :] = self.train_values[indices, :]

            last_index = last_iter_size

            if self.invalid_valid_ratio > 0:
                # random_entities = np.random.randint(
                #     0, self.nentity, last_index * self.invalid_valid_ratio)
                random_entities = np.random.choice(
                    self.entity_array, last_index * self.invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            # random_entities[current_index] = np.random.randint(
                            #     0, self.nentity)
                            random_entities[current_index] = np.random.choice(self.entity_array, 1)
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * \
                            (self.invalid_valid_ratio // 2) + \
                            (i * (self.invalid_valid_ratio // 2) + j)

                        while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            # random_entities[current_index] = np.random.randint(
                            #     0, self.nentity)
                            random_entities[current_index] = np.random.choice(self.entity_array, 1)
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                return self.batch_indices, self.batch_values

            return self.batch_indices, self.batch_values

    def get_iteration_batch_nhop(self, current_batch_indices, node_neighbors,batch_size):

        self.batch_indices = np.empty(
            (batch_size * (self.invalid_valid_ratio + 1), 4)).astype(np.int32)
        self.batch_values = np.empty(
            (batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)
        indices = random.sample(range(len(current_batch_indices)), batch_size)

        self.batch_indices[:batch_size,
                           :] = current_batch_indices[indices, :]
        self.batch_values[:batch_size,
                          :] = np.ones((batch_size, 1))

        last_index = batch_size

        if self.invalid_valid_ratio > 0:
            # random_entities = np.random.randint(
            #     0, self.nentity, last_index * self.invalid_valid_ratio)
            random_entities = np.random.choice(
                self.entity_array, last_index * self.invalid_valid_ratio)

            # Precopying the same valid indices from 0 to batch_size to rest
            # of the indices
            self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
            self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

            for i in range(last_index):
                for j in range(self.invalid_valid_ratio // 2):
                    current_index = i * (self.invalid_valid_ratio // 2) + j

                    self.batch_indices[last_index + current_index,
                                       0] = random_entities[current_index]
                    self.batch_values[last_index + current_index, :] = [0]

                for j in range(self.invalid_valid_ratio // 2):
                    current_index = last_index * \
                        (self.invalid_valid_ratio // 2) + \
                        (i * (self.invalid_valid_ratio // 2) + j)

                    self.batch_indices[last_index + current_index,
                                       3] = random_entities[current_index]
                    self.batch_values[last_index + current_index, :] = [0]

            return self.batch_indices, self.batch_values

        return self.batch_indices, self.batch_values

    def get_graph_degree(self):
        graph = {}
        all_tiples = torch.cat([self.train_adj_matrix[0].transpose(
            0, 1), self.train_adj_matrix[1].unsqueeze(1)], dim=1)
        out_dict ={}
        in_dict = {}
        for data in all_tiples:
            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()

            if(source not in graph.keys()):
                #graph[source] = {}
                #graph[source][target] = value
                out_dict[source] = 1
            else:
                #graph[source][target] = value
                out_dict[source] +=1
        print("Graph created")   # graph {source: { target: value, .. }, ..}
        sum_outdegree = sum(out_dict.values())
        print('out_degree:', out_dict)
        print('sum outdegree:',sum_outdegree)
        #sum_indegree = sum(in_dict.values())
        #print('indegree:', in_dict)
        #print('sum indegree:',sum_indegree)
        pdb.set_trace()
        return graph

    def get_graph(self):
        graph = {}
        all_tiples = torch.cat([self.train_adj_matrix[0].transpose(
            0, 1), self.train_adj_matrix[1].unsqueeze(1)], dim=1)

        for data in all_tiples:
            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()

            if(source not in graph.keys()):
                graph[source] = {}
                graph[source][target] = value
            else:
                graph[source][target] = value
        print("Graph created")   # graph {source: { target: value, .. }, ..}
        return graph

    def get_graph_modified(self):
        graph = {}
        all_triples = torch.cat([self.train_adj_matrix[0].transpose(0, 1), self.train_adj_matrix[1].unsqueeze(1)], dim=1)
        source_out = np.zeros(shape=len(all_triples))
        target_out = np.zeros(shape= len(all_triples))
        out_dict = {}

        for i, data in enumerate(all_triples):
            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()
            if (source not in graph.keys()):
                graph[source] = {}
                graph[source][target] = value
            else:
                graph[source][target] = value

            if (source not in out_dict.keys()):
                out_dict[source] = [1, [i], []]
            else:
                out_dict[source][0] += 1
                out_dict[source][1].append(i)
            if (target not in  out_dict.keys()):
                out_dict[target] = [0, [],[i]]
            else:
                out_dict[target][2].append(i)

        for value in out_dict.values():
            if value[0] == 0:   # out_degree =0
                target_out[value[2]] = 0.5
            elif len(value[2]) == 0 :  # in_degree = 0
                source_out[value[1]] = value[0]
            else:
                source_out[value[1]] = value[0]
                target_out[value[2]] = value[0]

        print("Graph created")  # graph {source: { target: value, .. }, ..}
        return graph, torch.from_numpy(source_out), torch.from_numpy(target_out)

    def get_graph_sampled(self, sampled_indices):
        graph = {}
        # all_tiples = torch.cat([self.train_adj_matrix[0].transpose(
        #     0, 1), self.train_adj_matrix[1].unsqueeze(1)], dim=1)

        for data in sampled_indices:
            source = data[1]
            target = data[0]
            value = data[2]

            if(source not in graph.keys()):
                graph[source] = {}
                graph[source][target] = value
            else:
                graph[source][target] = value
        print("Graph created")   # graph {source: { target: value, .. }, ..}
        return graph

    def samlping_RW1(self,num_iters_per_epoch, f):
        # moddified ripple walk sampler
        # very slow, takes around 70s for each iteration containing 1000 triples, and we have 16 million triples
        sampled_indices = self.train_indices
        for iter_num in range(num_iters_per_epoch):
            print('Sampling begins for iteration:{}\n'.format(iter_num))  # iterarion: 70s *262 = 18340 s
            f.write('Sampling begins for iteration:{}\n'.format(iter_num))
            start_time = time.time()
            # select initial node, and store it in the index_subgraph list
            neig_random = np.random.choice(self.unsampled_source)  # unsampled_entities: source entity of unsampled triples
            # the neighbor node set of the initial nodes
            add_neighbors = list(self.unsampled_graph[neig_random].keys())  # target entities
            neighbors = []
            sampled_n = 0
            while (1):
                if 0 < sampled_n < self.batch_size:  # judge if we need to select that much neighbors
                    if len(neighbors) == 0:
                        neig_random = np.random.choice(self.unsampled_source)
                    else:
                        neig_random = np.random.choice(neighbors)  # sample from the current cluster (subgraph)
                    #add_neighbors =  list(np.random.choice(list(self.unsampled_graph[neig_random].keys()),int(len(self.unsampled_graph[neig_random].keys())*0.8)))
                    add_neighbors = list(self.unsampled_graph[neig_random].keys())
                if sampled_n == self.batch_size:
                    break
                for target in add_neighbors:
                    if sampled_n < self.batch_size:
                        triple = (neig_random,self.unsampled_graph[neig_random][target], target)
                        #print(triple)
                        sampled_indices[iter_num * self.batch_size + sampled_n, :] = np.array(triple).astype(np.int32)
                        sampled_n += 1
                        del self.unsampled_graph[triple[0]][target]
                        #pdb.set_trace()
                        if len(self.unsampled_graph[triple[0]]) == 0:
                            # del self.unsampled_graph[neig_random]
                            self.unsampled_source.remove(triple[0])
                        if target in self.unsampled_source:
                            neighbors.append(target)
                        f.write('{}\n'.format(str(triple[0])+'\t'+str(triple[1])+'\t'+str(triple[2])))
                    else:
                        break
            print('Samlping finish for iteration {}! Time:{}\n'.format(
                iter_num ,time.time() - start_time))  # time for sampling one source entity and all its neighbors: 0.05 - 0.5s
            f.write('Samlping finish  for iteration {} ! Time:{}\n'.format( iter_num, time.time() - start_time))
        return sampled_indices

    def samlping_RW2(self,num_iters_per_epoch, f):
        # moddified ripple walk sampler
        # very slow, takes around 70s for each iteration containing 1000 triples, and we have 16 million triples
        sampled_indices = self.train_indices
        for iter_num in range(num_iters_per_epoch):
            print('Sampling begins for iteration:{}\n'.format(iter_num))  # iterarion: 70s *262 = 18340 s
            f.write('Sampling begins for iteration:{}\n'.format(iter_num))
            start_time = time.time()
            # select initial node, and store it in the index_subgraph list
            neig_random = np.random.choice(self.unsampled_source)  # unsampled_entities: source entity of unsampled triples
            # the neighbor node set of the initial nodes
            add_neighbors = list(self.graph[neig_random].keys())  # target entities
            neighbors = []
            sampled_n = 0
            while (1):
                if 0 < sampled_n < self.batch_size:  # judge if we need to select that much neighbors
                    neig_random = np.random.choice(neighbors)
                    if len(neighbors) ==0:
                        neig_random = np.random.choice(self.unsampled_source)
                    add_neighbors = list(np.random.choice(list(self.unsampled_graph[neig_random].keys()),int(len(self.unsampled_graph[neig_random].keys())*0.8)))
                if sampled_n == self.batch_size:
                    break
                for target in add_neighbors:
                    if sampled_n < self.batch_size:
                        triple = (neig_random,self.unsampled_graph[neig_random][target], target)
                        sampled_indices[iter_num * self.batch_size + sampled_n, :] = np.array(triple).astype(np.int32)
                        sampled_n += 1
                        # del self.unsampled_graph[triple[0]][target]
                        # if len(self.unsampled_graph[triple[0]]) == 0:
                        #     # del self.unsampled_graph[neig_random]
                        #     self.unsampled_source.remove(triple[0])
                        # if target in self.unsampled_source:
                        #     neighbors.append(target)
                        if target in self.unsampled_source:
                            neighbors.append(target)
                    else:
                        break
            print('Samlping finish for iteration {}! Time:{}\n'.format(
                iter_num ,time.time() - start_time))  # time for sampling one source entity and all its neighbors: 0.05 - 0.5s
            f.write('Samlping finish  for iteration {} ! Time:{}\n'.format( iter_num, time.time() - start_time))
        return sampled_indices


    def samlping_RW3(self, unsampled_graph, unsampled_source, iter_num):
        # modified ripple walk sampler
        # very slow, takes around 70s for each iteration containing 1000 triples, and we have 16 million triples

        print('Sampling begins for iteration:{}\n'.format(iter_num))
        start_time = time.time()
        if iter_num == None:  # last iteration
            sampled_n = 0
            for source in unsampled_source:
                for target in unsampled_graph[source].keys():
                    triple =(source, unsampled_graph[source][target], target)
                    self.batch_indices[sampled_n,:] = np.array(triple).astype(np.int32)
                    sampled_n += 1
        else:
            # select initial node, and store it in the index_subgraph list
            neig_random = np.random.choice(unsampled_source)  # unsampled_entities: source entity of unsampled triples
            # the neighbor node set of the initial nodes
            add_neighbors = list(unsampled_graph[neig_random].keys())  # target entities
            neighbors = []
            sampled_n = 0
            while (1):
                if 0 < sampled_n < self.batch_size:  # judge if we need to select that much neighbors
                    if len(neighbors) == 0:
                        neig_random = np.random.choice(unsampled_source)
                    else:
                        neig_random = np.random.choice(neighbors)  # sample from the current cluster (subgraph)
                    add_neighbors = list(unsampled_graph[neig_random].keys())
                if sampled_n == self.batch_size:
                    break
                for target in add_neighbors:
                    if sampled_n < self.batch_size:
                        triple = (neig_random, unsampled_graph[neig_random][target], target)
                        self.batch_indices[sampled_n, :] = np.array(triple).astype(np.int32)
                        sampled_n += 1
                        del unsampled_graph[triple[0]][target]
                        if len(unsampled_graph[triple[0]]) == 0:
                            # del self.unsampled_graph[neig_random]
                            unsampled_source.remove(triple[0])
                        if target in unsampled_source:
                            neighbors.append(target)
                    else:
                        break
        print('Samlping finish! Time:{}\n'.format(
            time.time() - start_time))  # time for sampling one source entity and all its neighbors: 0.05 - 0.5s
        return unsampled_graph, unsampled_source

    def sampled_iteration_batch(self, sampled_indices, iter_num):
        if (iter_num + 1) * self.batch_size <= len(sampled_indices):  # not the last iteration
            self.batch_indices = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)
            indices = range(self.batch_size * iter_num,
                            self.batch_size * (iter_num + 1))
            # self.batch_indices[:self.batch_size,
            #                    :] = self.train_indices[indices, :]  # train_indices-> train_triples : [(e1_id,r_id,e2_id),(), ...]

            #unsampled_graph, unsampled_source = self.samlping_RW2( unsampled_graph, unsampled_source, iter_num)  # update self.batch_indices
            self.batch_indices[:self.batch_size,:] = sampled_indices[indices, :]
            self.batch_values[:self.batch_size,:] = self.train_values[indices, :]

            last_index = self.batch_size
            if self.invalid_valid_ratio > 0:
                # random_entities = np.random.randint(
                #     0, self.nentity, last_index * self.invalid_valid_ratio)
                random_entities = np.random.choice(
                    self.entity_array, last_index * self.invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            # random_entities[current_index] = np.random.randint(
                            #     0, self.nentity)
                            random_entities[current_index] = np.random.choice(self.entity_array, 1)
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * \
                                        (self.invalid_valid_ratio // 2) + \
                                        (i * (self.invalid_valid_ratio // 2) + j)

                        while (self.batch_indices[last_index + current_index, 0],
                               self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            # random_entities[current_index] = np.random.randint(
                            #     0, self.nentity)
                            random_entities[current_index] = np.random.choice(self.entity_array, 1)
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                return self.batch_indices, self.batch_values

            return self.batch_indices, self.batch_values  # [ : valid triples for one batch, :invalid triples for replacing head entity,  : invalid for replacing tail entity]

        else:  # the last iteration
            last_iter_size = len(sampled_indices) - self.batch_size * iter_num
            self.batch_indices = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            len(sampled_indices))
            # self.batch_indices[:last_iter_size, :] = self.train_indices[indices, :]
            #unsampled_graph, unsampled_source = self.samlping_RW2(unsampled_graph, unsampled_source,None)
            self.batch_indices[:last_iter_size, :] = sampled_indices[indices, :]
            self.batch_values[:last_iter_size, :] = self.train_values[indices, :]

            last_index = last_iter_size
            if self.invalid_valid_ratio > 0:
                # random_entities = np.random.randint(
                #     0, self.nentity, last_index * self.invalid_valid_ratio)
                random_entities = np.random.choice(
                    self.entity_array, last_index * self.invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            # random_entities[current_index] = np.random.randint(
                            #     0, self.nentity)
                            random_entities[current_index] = np.random.choice(self.entity_array, 1)
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * \
                                        (self.invalid_valid_ratio // 2) + \
                                        (i * (self.invalid_valid_ratio // 2) + j)

                        while (self.batch_indices[last_index + current_index, 0],
                               self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            # random_entities[current_index] = np.random.randint(
                            #     0, self.nentity)
                            random_entities[current_index] = np.random.choice(self.entity_array, 1)
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                return self.batch_indices, self.batch_values

            return self.batch_indices, self.batch_values


    def bfs(self, graph, source, nbd_size=2):
        visit = {}
        distance = {}
        parent = {}
        distance_lengths = {}

        visit[source] = 1
        distance[source] = 0
        parent[source] = (-1, -1)

        q = queue.Queue()
        q.put((source, -1))

        while(not q.empty()):
            top = q.get()
            if top[0] in graph.keys():   #邻接表， O（E）,E= 27w+ / 邻接矩阵，0(N^2),N= 1.3w
                for target in graph[top[0]].keys():
                    if(target in visit.keys()):   #不能多个source 指向一个target , 但可以一个source指向多个target
                        continue
                    else:
                        q.put((target, graph[top[0]][target]))

                        distance[target] = distance[top[0]] + 1

                        visit[target] = 1
                        if distance[target] > 2:  # record and backtrack at most two-hops?
                            continue
                        parent[target] = (top[0], graph[top[0]][target])

                        if distance[target] not in distance_lengths.keys():
                            distance_lengths[distance[target]] = 1

        neighbors = {}
        for target in visit.keys():   #O(E)
            if(distance[target] != nbd_size):
                continue
            edges = [-1, parent[target][1]]
            relations = []
            entities = [target]
            temp = target
            while(parent[temp] != (-1, -1)):
                relations.append(parent[temp][1])
                entities.append(parent[temp][0])
                temp = parent[temp][0]

            if(distance[target] in neighbors.keys()):
                neighbors[distance[target]].append(
                    (tuple(relations), tuple(entities[:-1])))
            else:
                neighbors[distance[target]] = [
                    (tuple(relations), tuple(entities[:-1]))]

        return neighbors  # neighbors {distance:[((relations, ),(entities, )), ....]}

    def get_further_neighbors(self, nbd_size=2):  #时间久
        neighbors = {}
        start_time = time.time()
        print("length of graph keys is ", len(self.graph.keys()))  #1.3w
        for source in self.graph.keys():  # O(N*E) = 27w * 1.3W
            # st_time = time.time()
            temp_neighbors = self.bfs(self.graph, source, nbd_size)
            for distance in temp_neighbors.keys():
                if(source in neighbors.keys()):
                    if(distance in neighbors[source].keys()):
                        neighbors[source][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors[source][distance] = temp_neighbors[distance]
                else:
                    neighbors[source] = {}
                    neighbors[source][distance] = temp_neighbors[distance]  # neighbors {source:{distance:[((relations,),(entities, )),....]} }

        print("time taken ", time.time() - start_time)

        print("length of neighbors dict is ", len(neighbors))
        return neighbors

    def get_batch_nhop_neighbors_all(self, args, batch_sources, node_neighbors, nbd_size=2):
        batch_source_triples = []
        print("length of unique_entities ", len(batch_sources))
        count = 0

        for source in batch_sources:
            # randomly select from the list of neighbors
            if source in node_neighbors.keys():
                nhop_list = node_neighbors[source][nbd_size]

                for i, tup in enumerate(nhop_list):
                    if(args.partial_2hop and i >= 2):
                        break

                    count += 1
                    batch_source_triples.append([source, nhop_list[i][0][-1], nhop_list[i][0][0],
                                                 nhop_list[i][1][0]])  # [source entity, relation between source_entity , first relation between target_entity, target entity]

        return np.array(batch_source_triples).astype(np.int32)  #[[source, relation_source, relation_target, target],[], ...]
    def get_unique_entites_number(self, batch_triples):
        unique_entities = set()
        for i in range(len(batch_triples)):
            e1, relation, e2 = batch_triples[i][0], batch_triples[i][1], batch_triples[i][2]
            unique_entities.add(e1)
            unique_entities.add(e2)
        return len(unique_entities)

    def transe_scoring(self, batch_inputs, entity_embeddings, relation_embeddings):
        source_embeds = entity_embeddings[batch_inputs[:, 0]]
        relation_embeds = relation_embeddings[batch_inputs[:, 1]]
        tail_embeds = entity_embeddings[batch_inputs[:, 2]]
        x = source_embeds + relation_embeds - tail_embeds
        x = torch.norm(x, p=1, dim=1)
        return x

    def get_validation_pred(self, args, model, unique_entities):     # unique_entities for training set
        average_hits_at_100_head, average_hits_at_100_tail = [], []
        average_hits_at_ten_head, average_hits_at_ten_tail = [], []
        average_hits_at_three_head, average_hits_at_three_tail = [], []
        average_hits_at_one_head, average_hits_at_one_tail = [], []
        average_mean_rank_head, average_mean_rank_tail = [], []
        average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []

        for iters in range(1):
            start_time = time.time()

            indices = [i for i in range(len(self.test_indices))]
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(self.test_indices))
            entity_list = [j for i, j in self.entity2id.items()]

            ranks_head, ranks_tail = [], []
            reciprocal_ranks_head, reciprocal_ranks_tail = [], []
            hits_at_100_head, hits_at_100_tail = 0, 0
            hits_at_ten_head, hits_at_ten_tail = 0, 0
            hits_at_three_head, hits_at_three_tail = 0, 0
            hits_at_one_head, hits_at_one_tail = 0, 0

            for i in range(batch_indices.shape[0]):
                print(len(ranks_head))
                start_time_it = time.time()
                new_x_batch_head = np.tile(
                    batch_indices[i, :], (len(self.entity2id), 1))
                new_x_batch_tail = np.tile(
                    batch_indices[i, :], (len(self.entity2id), 1))

                if(batch_indices[i, 0] not in unique_entities or batch_indices[i, 2] not in unique_entities):
                    continue

                new_x_batch_head[:, 0] = entity_list
                new_x_batch_tail[:, 2] = entity_list

                last_index_head = []  # array of already existing triples
                last_index_tail = []
                for tmp_index in range(len(new_x_batch_head)):
                    temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1],
                                        new_x_batch_head[tmp_index][2])
                    if temp_triple_head in self.valid_triples_dict.keys():
                        last_index_head.append(tmp_index)

                    temp_triple_tail = (new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][1],
                                        new_x_batch_tail[tmp_index][2])
                    if temp_triple_tail in self.valid_triples_dict.keys():
                        last_index_tail.append(tmp_index)

                # Deleting already existing triples, leftover triples are invalid, according
                # to train, validation and test data
                # Note, all of them maynot be actually invalid
                new_x_batch_head = np.delete(
                    new_x_batch_head, last_index_head, axis=0)
                new_x_batch_tail = np.delete(
                    new_x_batch_tail, last_index_tail, axis=0)

                # adding the current valid triples to the top, i.e, index 0
                new_x_batch_head = np.insert(
                    new_x_batch_head, 0, batch_indices[i], axis=0)
                new_x_batch_tail = np.insert(
                    new_x_batch_tail, 0, batch_indices[i], axis=0)

                import math
                # Have to do this, because it doesn't fit in memory

                #if 'WN' in args.data:
                #    num_triples_each_shot = int(
                #        math.ceil(new_x_batch_head.shape[0] / 4))
                
                #    scores1_head = model.batch_test(torch.LongTensor(
                #        new_x_batch_head[:num_triples_each_shot, :]).cuda())
                #    scores2_head = model.batch_test(torch.LongTensor(
                #        new_x_batch_head[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                #    scores3_head = model.batch_test(torch.LongTensor(
                #        new_x_batch_head[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                #    scores4_head = model.batch_test(torch.LongTensor(
                #        new_x_batch_head[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())
                    # scores5_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[4 * num_triples_each_shot: 5 * num_triples_each_shot, :]).cuda())
                    # scores6_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[5 * num_triples_each_shot: 6 * num_triples_each_shot, :]).cuda())
                    # scores7_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[6 * num_triples_each_shot: 7 * num_triples_each_shot, :]).cuda())
                    # scores8_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[7 * num_triples_each_shot: 8 * num_triples_each_shot, :]).cuda())
                    # scores9_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[8 * num_triples_each_shot: 9 * num_triples_each_shot, :]).cuda())
                    # scores10_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[9 * num_triples_each_shot:, :]).cuda())

               #     scores_head = torch.cat(
               #         [scores1_head, scores2_head, scores3_head, scores4_head], dim=0)
                    #scores5_head, scores6_head, scores7_head, scores8_head,
                    # cores9_head, scores10_head], dim=0)
                #else:
                scores_head = model.batch_test(new_x_batch_head)

                sorted_scores_head, sorted_indices_head = torch.sort(
                    scores_head.view(-1), dim=-1, descending=True)
                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_head.append(
                    np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])

                # Tail part here

                #if 'WN' in args.data:
                #    num_triples_each_shot = int(
                #        math.ceil(new_x_batch_tail.shape[0] / 4))

                #    scores1_tail = model.batch_test(torch.LongTensor(
                #        new_x_batch_tail[:num_triples_each_shot, :]).cuda())
                #    scores2_tail = model.batch_test(torch.LongTensor(
                #        new_x_batch_tail[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                #    scores3_tail = model.batch_test(torch.LongTensor(
                #        new_x_batch_tail[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                #    scores4_tail = model.batch_test(torch.LongTensor(
                #        new_x_batch_tail[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())
                    # scores5_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[4 * num_triples_each_shot: 5 * num_triples_each_shot, :]).cuda())
                    # scores6_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[5 * num_triples_each_shot: 6 * num_triples_each_shot, :]).cuda())
                    # scores7_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[6 * num_triples_each_shot: 7 * num_triples_each_shot, :]).cuda())
                    # scores8_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[7 * num_triples_each_shot: 8 * num_triples_each_shot, :]).cuda())
                    # scores9_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[8 * num_triples_each_shot: 9 * num_triples_each_shot, :]).cuda())
                    # scores10_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[9 * num_triples_each_shot:, :]).cuda())

                #    scores_tail = torch.cat(
                #        [scores1_tail, scores2_tail, scores3_tail, scores4_tail], dim=0)
                    #     scores5_tail, scores6_tail, scores7_tail, scores8_tail,
                    #     scores9_tail, scores10_tail], dim=0)

                #else:
                scores_tail = model.batch_test(new_x_batch_tail)

                sorted_scores_tail, sorted_indices_tail = torch.sort(
                    scores_tail.view(-1), dim=-1, descending=True)

                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_tail.append(
                    np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
                # print("sample - ", ranks_head[-1], ranks_tail[-1])
                if i% 10000 == 0:
                    print("sample 10000 triples -> time: {}", time.time()-start_time_it)
                    start_time_it = time.time()

            for i in range(len(ranks_head)):
                if ranks_head[i] <= 100:
                    hits_at_100_head = hits_at_100_head + 1
                if ranks_head[i] <= 10:
                    hits_at_ten_head = hits_at_ten_head + 1
                if ranks_head[i] <= 3:
                    hits_at_three_head = hits_at_three_head + 1
                if ranks_head[i] == 1:
                    hits_at_one_head = hits_at_one_head + 1

            for i in range(len(ranks_tail)):
                if ranks_tail[i] <= 100:
                    hits_at_100_tail = hits_at_100_tail + 1
                if ranks_tail[i] <= 10:
                    hits_at_ten_tail = hits_at_ten_tail + 1
                if ranks_tail[i] <= 3:
                    hits_at_three_tail = hits_at_three_tail + 1
                if ranks_tail[i] == 1:
                    hits_at_one_tail = hits_at_one_tail + 1

            assert len(ranks_head) == len(reciprocal_ranks_head)
            assert len(ranks_tail) == len(reciprocal_ranks_tail)
            print("here {}".format(len(ranks_head)))
            print("\nCurrent iteration time {}".format(time.time() - start_time))
            print("Stats for replacing head are -> ")
            print("Current iteration Hits@100 are {}".format(
                hits_at_100_head / float(len(ranks_head))))
            print("Current iteration Hits@10 are {}".format(
                hits_at_ten_head / len(ranks_head)))
            print("Current iteration Hits@3 are {}".format(
                hits_at_three_head / len(ranks_head)))
            print("Current iteration Hits@1 are {}".format(
                hits_at_one_head / len(ranks_head)))
            print("Current iteration Mean rank {}".format(
                sum(ranks_head) / len(ranks_head)))
            print("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)))

            print("\nStats for replacing tail are -> ")
            print("Current iteration Hits@100 are {}".format(
                hits_at_100_tail / len(ranks_head)))
            print("Current iteration Hits@10 are {}".format(
                hits_at_ten_tail / len(ranks_head)))
            print("Current iteration Hits@3 are {}".format(
                hits_at_three_tail / len(ranks_head)))
            print("Current iteration Hits@1 are {}".format(
                hits_at_one_tail / len(ranks_head)))
            print("Current iteration Mean rank {}".format(
                sum(ranks_tail) / len(ranks_tail)))
            print("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)))

            average_hits_at_100_head.append(
                hits_at_100_head / len(ranks_head))
            average_hits_at_ten_head.append(
                hits_at_ten_head / len(ranks_head))
            average_hits_at_three_head.append(
                hits_at_three_head / len(ranks_head))
            average_hits_at_one_head.append(
                hits_at_one_head / len(ranks_head))
            average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
            average_mean_recip_rank_head.append(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

            average_hits_at_100_tail.append(
                hits_at_100_tail / len(ranks_head))
            average_hits_at_ten_tail.append(
                hits_at_ten_tail / len(ranks_head))
            average_hits_at_three_tail.append(
                hits_at_three_tail / len(ranks_head))
            average_hits_at_one_tail.append(
                hits_at_one_tail / len(ranks_head))
            average_mean_rank_tail.append(sum(ranks_tail) / len(ranks_tail))
            average_mean_recip_rank_tail.append(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail))

        print("\nAveraged stats for replacing head are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_head) / len(average_hits_at_100_head)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_head) / len(average_hits_at_three_head)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_head) / len(average_hits_at_one_head)))
        print("Mean rank {}".format(
            sum(average_mean_rank_head) / len(average_mean_rank_head)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head)))

        print("\nAveraged stats for replacing tail are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)))
        print("Mean rank {}".format(
            sum(average_mean_rank_tail) / len(average_mean_rank_tail)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)))

        cumulative_hits_100 = (sum(average_hits_at_100_head) / len(average_hits_at_100_head)
                               + sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) / 2
        cumulative_hits_ten = (sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)
                               + sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) / 2
        cumulative_hits_three = (sum(average_hits_at_three_head) / len(average_hits_at_three_head)
                                 + sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) / 2
        cumulative_hits_one = (sum(average_hits_at_one_head) / len(average_hits_at_one_head)
                               + sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) / 2
        cumulative_mean_rank = (sum(average_mean_rank_head) / len(average_mean_rank_head)
                                + sum(average_mean_rank_tail) / len(average_mean_rank_tail)) / 2
        cumulative_mean_recip_rank = (sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head) + sum(
            average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) / 2

        print("\nCumulative stats are -> ")
        print("Hits@100 are {}".format(cumulative_hits_100))
        print("Hits@10 are {}".format(cumulative_hits_ten))
        print("Hits@3 are {}".format(cumulative_hits_three))
        print("Hits@1 are {}".format(cumulative_hits_one))
        print("Mean rank {}".format(cumulative_mean_rank))
        print("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))

    def get_validation_pred_modified(self, args, model, unique_entities, f):     # unique_entities for training set
        average_hits_at_100_head, average_hits_at_100_tail = [], []
        average_hits_at_ten_head, average_hits_at_ten_tail = [], []
        average_hits_at_three_head, average_hits_at_three_tail = [], []
        average_hits_at_one_head, average_hits_at_one_tail = [], []
        average_mean_rank_head, average_mean_rank_tail = [], []
        average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []

        for iters in range(1):
            start_time = time.time()

            indices = [i for i in range(len(self.test_indices))]
            batch_indices = self.test_indices[indices, :]
            print("Sampled indices")
            print("test set length ", len(self.test_indices))
            f.write("Sampled indices\n")
            f.write("test set length {} \n".format(len(self.test_indices)))

            ranks_head, ranks_tail = [], []
            reciprocal_ranks_head, reciprocal_ranks_tail = [], []
            hits_at_100_head, hits_at_100_tail = 0, 0
            hits_at_ten_head, hits_at_ten_tail = 0, 0
            hits_at_three_head, hits_at_three_tail = 0, 0
            hits_at_one_head, hits_at_one_tail = 0, 0
            
            #strat_time_it =time.time()
            for i in range(batch_indices.shape[0]):
                #print(len(ranks_head))
                if i == 0:
                    start_time_it = time.time()
                new_x_batch_head = np.array([(self.test_data[0][i][1][j], self.test_data[0][i][2], self.test_data[0][i][3]) \
                                             for j in range(len(self.test_data[0][i][1]))]).astype(np.int32)
                new_x_batch_tail = np.array([(self.test_data[0][i][0], self.test_data[0][i][2], self.test_data[0][i][4][j]) \
                                             for j in range(len(self.test_data[0][i][4]))]).astype(np.int32)

                if(batch_indices[i, 0] not in unique_entities or batch_indices[i, 2] not in unique_entities):
                    continue

                # adding the current valid triples to the top, i.e, index 0
                new_x_batch_head = np.insert(
                    new_x_batch_head, 0, batch_indices[i], axis=0)
                new_x_batch_tail = np.insert(
                    new_x_batch_tail, 0, batch_indices[i], axis=0)

                import math
                # Have to do this, because it doesn't fit in memory

                #if 'WN' in args.data:
                #    num_triples_each_shot = int(
                #        math.ceil(new_x_batch_head.shape[0] / 4))

                #    scores1_head = model.batch_test(torch.LongTensor(
                #        new_x_batch_head[:num_triples_each_shot, :]).cuda())
                #    scores2_head = model.batch_test(torch.LongTensor(
                #        new_x_batch_head[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                #    scores3_head = model.batch_test(torch.LongTensor(
                #        new_x_batch_head[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                #    scores4_head = model.batch_test(torch.LongTensor(
                #        new_x_batch_head[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())
                    # scores5_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[4 * num_triples_each_shot: 5 * num_triples_each_shot, :]).cuda())
                    # scores6_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[5 * num_triples_each_shot: 6 * num_triples_each_shot, :]).cuda())
                    # scores7_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[6 * num_triples_each_shot: 7 * num_triples_each_shot, :]).cuda())
                    # scores8_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[7 * num_triples_each_shot: 8 * num_triples_each_shot, :]).cuda())
                    # scores9_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[8 * num_triples_each_shot: 9 * num_triples_each_shot, :]).cuda())
                    # scores10_head = model.batch_test(torch.LongTensor(
                    #     new_x_batch_head[9 * num_triples_each_shot:, :]).cuda())

                #    scores_head = torch.cat(
                #        [scores1_head, scores2_head, scores3_head, scores4_head], dim=0)
                    #scores5_head, scores6_head, scores7_head, scores8_head,
                    # cores9_head, scores10_head], dim=0)
                #else:
                scores_head = model.batch_test(new_x_batch_head)

                sorted_scores_head, sorted_indices_head = torch.sort(
                    scores_head.view(-1), dim=-1, descending=True)
                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_head.append(
                    np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])

                # Tail part here

                #if 'WN' in args.data:
                #    num_triples_each_shot = int(
                #        math.ceil(new_x_batch_tail.shape[0] / 4))

                #    scores1_tail = model.batch_test(torch.LongTensor(
                #        new_x_batch_tail[:num_triples_each_shot, :]).cuda())
                #    scores2_tail = model.batch_test(torch.LongTensor(
                #       new_x_batch_tail[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                #    scores3_tail = model.batch_test(torch.LongTensor(
                #        new_x_batch_tail[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                #    scores4_tail = model.batch_test(torch.LongTensor(
                #        new_x_batch_tail[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())
                    # scores5_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[4 * num_triples_each_shot: 5 * num_triples_each_shot, :]).cuda())
                    # scores6_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[5 * num_triples_each_shot: 6 * num_triples_each_shot, :]).cuda())
                    # scores7_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[6 * num_triples_each_shot: 7 * num_triples_each_shot, :]).cuda())
                    # scores8_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[7 * num_triples_each_shot: 8 * num_triples_each_shot, :]).cuda())
                    # scores9_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[8 * num_triples_each_shot: 9 * num_triples_each_shot, :]).cuda())
                    # scores10_tail = model.batch_test(torch.LongTensor(
                    #     new_x_batch_tail[9 * num_triples_each_shot:, :]).cuda())

                #    scores_tail = torch.cat(
                #        [scores1_tail, scores2_tail, scores3_tail, scores4_tail], dim=0)
                    #     scores5_tail, scores6_tail, scores7_tail, scores8_tail,
                    #     scores9_tail, scores10_tail], dim=0)

                #else:
                scores_tail = model.batch_test(new_x_batch_tail)

                sorted_scores_tail, sorted_indices_tail = torch.sort(
                    scores_tail.view(-1), dim=-1, descending=True)

                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_tail.append(
                    np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
                #print("sample - ", ranks_head[-1], ranks_tail[-1])
                if i% 10000 == 0:
                    print("sample 10000 triples for {} times -> time: {}".format(i//10000, time.time()-start_time_it))
                    start_time_it = time.time()

            for i in range(len(ranks_head)):
                if ranks_head[i] <= 100:
                    hits_at_100_head = hits_at_100_head + 1
                if ranks_head[i] <= 10:
                    hits_at_ten_head = hits_at_ten_head + 1
                if ranks_head[i] <= 3:
                    hits_at_three_head = hits_at_three_head + 1
                if ranks_head[i] == 1:
                    hits_at_one_head = hits_at_one_head + 1

            for i in range(len(ranks_tail)):
                if ranks_tail[i] <= 100:
                    hits_at_100_tail = hits_at_100_tail + 1
                if ranks_tail[i] <= 10:
                    hits_at_ten_tail = hits_at_ten_tail + 1
                if ranks_tail[i] <= 3:
                    hits_at_three_tail = hits_at_three_tail + 1
                if ranks_tail[i] == 1:
                    hits_at_one_tail = hits_at_one_tail + 1

            assert len(ranks_head) == len(reciprocal_ranks_head)
            assert len(ranks_tail) == len(reciprocal_ranks_tail)
            print("here {}".format(len(ranks_head)))
            print("\nCurrent iteration time {}".format(time.time() - start_time))
            f.write("\nCurrent iteration time {}".format(time.time() - start_time))
            print("Stats for replacing head are -> ")
            f.write("Stats for replacing head are -> ")
            print("Current iteration Hits@100 are {}".format(
                hits_at_100_head / float(len(ranks_head))))
            print("Current iteration Hits@10 are {}".format(
                hits_at_ten_head / len(ranks_head)))
            print("Current iteration Hits@3 are {}".format(
                hits_at_three_head / len(ranks_head)))
            print("Current iteration Hits@1 are {}".format(
                hits_at_one_head / len(ranks_head)))
            print("Current iteration Mean rank {}".format(
                sum(ranks_head) / len(ranks_head)))
            print("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)))
            f.write("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)))



            print("\nStats for replacing tail are -> ")
            f.write("\nStats for replacing tail are -> ")
            print("Current iteration Hits@100 are {}".format(
                hits_at_100_tail / len(ranks_head)))
            print("Current iteration Hits@10 are {}".format(
                hits_at_ten_tail / len(ranks_head)))
            print("Current iteration Hits@3 are {}".format(
                hits_at_three_tail / len(ranks_head)))
            print("Current iteration Hits@1 are {}".format(
                hits_at_one_tail / len(ranks_head)))
            print("Current iteration Mean rank {}".format(
                sum(ranks_tail) / len(ranks_tail)))
            print("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)))
            f.write("Current iteration Mean Reciprocal Rank {}".format(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)))

            average_hits_at_100_head.append(
                hits_at_100_head / len(ranks_head))
            average_hits_at_ten_head.append(
                hits_at_ten_head / len(ranks_head))
            average_hits_at_three_head.append(
                hits_at_three_head / len(ranks_head))
            average_hits_at_one_head.append(
                hits_at_one_head / len(ranks_head))
            average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
            average_mean_recip_rank_head.append(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

            average_hits_at_100_tail.append(
                hits_at_100_tail / len(ranks_head))
            average_hits_at_ten_tail.append(
                hits_at_ten_tail / len(ranks_head))
            average_hits_at_three_tail.append(
                hits_at_three_tail / len(ranks_head))
            average_hits_at_one_tail.append(
                hits_at_one_tail / len(ranks_head))
            average_mean_rank_tail.append(sum(ranks_tail) / len(ranks_tail))
            average_mean_recip_rank_tail.append(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail))

        print("\nAveraged stats for replacing head are -> ")
        f.write("\nAveraged stats for replacing head are -> \n")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_head) / len(average_hits_at_100_head)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_head) / len(average_hits_at_three_head)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_head) / len(average_hits_at_one_head)))
        print("Mean rank {}".format(
            sum(average_mean_rank_head) / len(average_mean_rank_head)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head)))
        f.write("Mean Reciprocal Rank {}\n".format(
            sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head)))

        print("\nAveraged stats for replacing tail are -> ")
        f.write("\nAveraged stats for replacing tail are -> ")
        print("Hits@100 are {}".format(
            sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)))
        print("Hits@10 are {}".format(
            sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)))
        print("Hits@3 are {}".format(
            sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)))
        print("Hits@1 are {}".format(
            sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)))
        print("Mean rank {}".format(
            sum(average_mean_rank_tail) / len(average_mean_rank_tail)))
        print("Mean Reciprocal Rank {}".format(
            sum(average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)))
        f.write("Mean Reciprocal Rank {}\n".format(
            sum(average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)))

        cumulative_hits_100 = (sum(average_hits_at_100_head) / len(average_hits_at_100_head)
                               + sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) / 2
        cumulative_hits_ten = (sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)
                               + sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) / 2
        cumulative_hits_three = (sum(average_hits_at_three_head) / len(average_hits_at_three_head)
                                 + sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) / 2
        cumulative_hits_one = (sum(average_hits_at_one_head) / len(average_hits_at_one_head)
                               + sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) / 2
        cumulative_mean_rank = (sum(average_mean_rank_head) / len(average_mean_rank_head)
                                + sum(average_mean_rank_tail) / len(average_mean_rank_tail)) / 2
        cumulative_mean_recip_rank = (sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head) + sum(
            average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) / 2

        print("\nCumulative stats are -> ")
        f.write("\nCumulative stats are -> \n")
        print("Hits@100 are {}".format(cumulative_hits_100))
        print("Hits@10 are {}".format(cumulative_hits_ten))
        print("Hits@3 are {}".format(cumulative_hits_three))
        print("Hits@1 are {}".format(cumulative_hits_one))
        print("Mean rank {}".format(cumulative_mean_rank))
        print("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))
        f.write("Mean Reciprocal Rank {}\n".format(cumulative_mean_recip_rank))
