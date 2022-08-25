import torch
import os
import numpy as np


def read_entity_from_id(filename='./data/WN18RR/entity2id.txt'):
    entity2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                entity, entity_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id


def read_relation_from_id(filename='./data/WN18RR/relation2id.txt'):
    relation2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                relation, relation_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id


def init_embeddings(entity_file, relation_file):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            entity_emb.append([float(val) for val in line.strip().split()])

    with open(relation_file) as f:
        for line in f:
            relation_emb.append([float(val) for val in line.strip().split()])

    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)


def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2


def load_data(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()

    # this is list for relation triples
    triples_data = []

    # for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
    # columnn of sparse tensor, data contains the type of relation
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with
    rows, cols, data = [], [], []
    unique_entities = set()
    for line in lines:
        e1, relation, e2 = parse_line(line)
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append(
            (entity2id[e1], relation2id[relation], entity2id[e2]))
        if not directed:
                # Connecting source and tail entity
            rows.append(entity2id[e1])
            cols.append(entity2id[e2])
            if is_unweigted:
                data.append(1)
            else:
                data.append(relation2id[relation])

        # Connecting tail and source entity
        rows.append(entity2id[e2])
        cols.append(entity2id[e1])
        if is_unweigted:
            data.append(1)
        else:
            data.append(relation2id[relation])

    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data), list(unique_entities)

def load_data_modified(split_dict,data_type, is_unweigted=False, directed=True):
    # this is list for relation triples
    triples_data = []

    # for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
    # columnn of sparse tensor, data contains the type of relation
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with
    unique_entities = set()

    triples_dict = split_dict[data_type]
    if data_type == 'train':
        for i in range(len(triples_dict['head'])):    # train triples: 16,109,182
            e1, relation, e2 = triples_dict['head'][i], triples_dict['relation'][i],triples_dict['tail'][i]
            unique_entities.add(e1)
            unique_entities.add(e2)
            triples_data.append((e1, relation, e2))
    else:                                               # valid triples: 429,456 , test: 598,543
        for i in range(len(triples_dict['head'])):     # shape of  head_neg: ndarray(num of valid/test triples, 500)
            e1, e1_n, relation, e2, e2_n = triples_dict['head'][i],triples_dict['head_neg'][i], triples_dict['relation'][i],triples_dict['tail'][i],triples_dict['tail_neg'][i]
            unique_entities.add(e1)
            unique_entities.add(e2)
            triples_data.append((e1, e1_n, relation, e2, e2_n))
    # Connecting tail and source entity
    rows = list(triples_dict['tail'])  # ndarray->list ,  append valid triples
    cols = list(triples_dict['head'])
    if is_unweigted:
        data = [1 for i in range(len(triples_dict['head']))]
    else:
        data = list(triples_dict['relation'])
    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data), list(unique_entities)


def build_data(path='./data/WN18RR/', is_unweigted=False, directed=True):
    entity2id = read_entity_from_id(path + 'entity2id.txt')  # dict{entity:id, }
    relation2id = read_relation_from_id(path + 'relation2id.txt')

    # Adjacency matrix only required for training phase
    # Currenlty creating as unweighted, undirected
    train_triples, train_adjacency_mat, unique_entities_train = load_data(os.path.join(
        path, 'train.txt'), entity2id, relation2id, is_unweigted, directed)
    validation_triples, valid_adjacency_mat, unique_entities_validation = load_data(
        os.path.join(path, 'valid.txt'), entity2id, relation2id, is_unweigted, directed)
    test_triples, test_adjacency_mat, unique_entities_test = load_data(os.path.join(
        path, 'test.txt'), entity2id, relation2id, is_unweigted, directed)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    left_entity, right_entity = {}, {}

    with open(os.path.join(path, 'train.txt')) as f:
        lines = f.readlines()

    for line in lines:
        e1, relation, e2 = parse_line(line)

        # Count number of occurences for each (e1, relation)
        if relation2id[relation] not in left_entity:
            left_entity[relation2id[relation]] = {}
        if entity2id[e1] not in left_entity[relation2id[relation]]:
            left_entity[relation2id[relation]][entity2id[e1]] = 0
        left_entity[relation2id[relation]][entity2id[e1]] += 1  #{relation_id: {entity1_id: , }}

        # Count number of occurences for each (relation, e2)
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1

    left_entity_avg = {}
    for i in range(len(relation2id)):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i])  #average left_entity degree for each relation

    right_entity_avg = {}
    for i in range(len(relation2id)):
        right_entity_avg[i] = sum(
            right_entity[i].values()) * 1.0 / len(right_entity[i])  #average right_entity degree for each relation

    headTailSelector = {}
    for i in range(len(relation2id)):    # for train data: average ratio of right_entity degree for each relation
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i])

    return (train_triples, train_adjacency_mat), (validation_triples, valid_adjacency_mat), (test_triples, test_adjacency_mat), \
        entity2id, relation2id, headTailSelector, unique_entities_train
   # ( [(e1_id,r_id,e2_id),(), ...],([e2_id,..],[e1_id,..],[r_id,...]) )
   # {e:e_id}, {r:r_id}, {r_id: number,   }, [e,...]


def build_data_modified(split_dict, relation_array,is_unweigted=False, directed=True):

    train_triples, train_adjacency_mat, unique_entities_train = load_data_modified(split_dict, 'train',  is_unweigted, directed)
    validation_triples, valid_adjacency_mat, unique_entities_validation = load_data_modified(split_dict,'valid', is_unweigted, directed)
    test_triples, test_adjacency_mat, unique_entities_test = load_data_modified(split_dict, 'test', is_unweigted, directed)

    left_entity, right_entity = {}, {}

    train_triples_dict = split_dict['train']
    for i in range(len(train_triples_dict['head'])):  # train triples: 16,109,182
        e1, relation, e2 = train_triples_dict['head'][i], train_triples_dict['relation'][i], train_triples_dict['tail'][i]
        # Count number of occurences for each (e1, relation)
        if relation not in left_entity:
            left_entity[relation] = {}
        if e1 not in left_entity[relation]:
            left_entity[relation][e1] = 0
        left_entity[relation][e1] += 1  #{relation_id: {entity1_id: , }}

        # Count number of occurences for each (relation, e2)
        if relation not in right_entity:
            right_entity[relation] = {}
        if e2 not in right_entity[relation]:
            right_entity[relation][e2] = 0
        right_entity[relation][e2] += 1

    left_entity_avg = {}
    for i in relation_array:  # nrelation=535
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i])  #average left_entity degree for each relation

    right_entity_avg = {}
    for i in relation_array:
        right_entity_avg[i] = sum(
            right_entity[i].values()) * 1.0 / len(right_entity[i])  #average right_entity degree for each relation

    headTailSelector = {}
    for i in relation_array:    # for train data: average ratio of target_entity degree for each relation
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i])

    return (train_triples, train_adjacency_mat), (validation_triples, valid_adjacency_mat), (test_triples, test_adjacency_mat), \
            headTailSelector, unique_entities_train
   # ( [(e1_id,r_id,e2_id),(), ...],([e2_id,..],[e1_id,..],[r_id,...]) )
   # {r_id: number,   }, [e_id,...]