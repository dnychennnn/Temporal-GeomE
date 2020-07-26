#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 18:17:51 2019

@author: cjxu
"""
import torch
from torch import nn
import numpy as np

from datasets import TemporalDataset
from models import ComplEx, TComplEx, TNTComplEx, TGeomE1, TGeomE2, TGeomE3


import os
import pickle
import pkg_resources
from pathlib import Path
from typing import Dict

data_dir = 'yago12k'
dataset = TemporalDataset(data_dir)


#scores = {}
#scores['rhs'] = torch.zeros(len(test),n_entities).float().cuda()
#scores['lhs'] = torch.zeros(len(test),n_entities).float().cuda()
#targets = {}
#targets['rhs'] = torch.zeros(len(test), 1).float().cuda()
#targets['lhs'] = torch.zeros(len(test), 1).float().cuda()

scores = {}
scores['rhs'] = torch.zeros(len(dataset.data['test']),dataset.n_entities).float()#.cuda()
scores['lhs'] = torch.zeros(len(dataset.data['test']),dataset.n_entities).float()#.cuda()
targets = {}
targets['rhs'] = torch.zeros(len(dataset.data['test']), 1).float()#.cuda()
targets['lhs'] = torch.zeros(len(dataset.data['test']), 1).float()#.cuda()


missing = ['rhs', 'lhs']
mean_reciprocal_rank = {}
mean_rank = {}
hits_at = {}
at = (1, 3, 10)

batch_size = 1
component={}
device = 'cuda'

num=1

i=0
component[i]={}
component[i]['model']='TGeomE2'
component[i]['rank']=2000
component[i]['lr']=0.1
component[i]['batch']=50
component[i]['emb_reg']=0.05
component[i]['time_reg']=0.0025
component[i]['time_granularity']=0
component[i]['epoch_pretrain']=0


i+=1
component[i]={}
component[i]['model']='TGeomE3'
component[i]['rank']=2000
component[i]['lr']=0.1
component[i]['batch']=50
component[i]['emb_reg']=0.05
component[i]['time_reg']=0.005
component[i]['time_granularity']=1
component[i]['epoch_pretrain']=50


###############  load models one by one #######
for i in range(num):
    
    root = 'results/'+ data_dir +'/' + component[i]['model']
    PATH=os.path.join(root,'rank{:.0f}/lr{:.4f}/batch{:.0f}/time_granularity{:02d}/emb_reg{:.5f}/time_reg{:.5f}/epoch_pretrain{:.5f}'.format(component[i]['rank'],
                      component[i]['lr'],component[i]['batch'],component[i]['time_granularity'], component[i]['emb_reg'], component[i]['time_reg'], component[i]['epoch_pretrain']))

    
    init= 1e-3
    sizes = dataset.get_shape()
    model = {
        'ComplEx': ComplEx(sizes, component[i]['rank']),
        'TComplEx': TComplEx(sizes, component[i]['rank'], no_time_emb=False),
        'TNTComplEx': TNTComplEx(sizes, component[i]['rank'], no_time_emb=False),
        'TGeomE1': TGeomE1(sizes, component[i]['rank'], no_time_emb=False, time_granularity=component[i]['time_granularity']),
        'TGeomE2': TGeomE2(sizes, component[i]['rank'], no_time_emb=False, time_granularity=component[i]['time_granularity']),
        'TGeomE3': TGeomE3(sizes, component[i]['rank'], no_time_emb=False, time_granularity=component[i]['time_granularity'])
    }[component[i]['model']]
    
    model.load_state_dict(torch.load(os.path.join(PATH, component[i]['model'] +'.pkl')))
    model.to(device)
    print('model{:.0f}'.format(i)+' loaded')


    test = dataset.get_examples('test')


    missing = ['rhs', 'lhs']

    mean_reciprocal_rank = {}
    hits_at = {}


###############  calculate scores of facts from each model #######   
    for m in missing:
        if dataset.interval:## for datasets YAGO11k and wikidata12k, q is numpy.array(str)
            year2id = dataset.time_dict
            examples = test
            q = np.copy(examples)
            if m == 'lhs':
                tmp = np.copy(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] = q[:, 1].astype('uint64')+dataset.n_predicates // 2

        else:
            examples = torch.from_numpy(test.astype('int64')).cuda()
            q = examples.clone()
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += dataset.n_predicates // 2

    
        queries = q
        filters = dataset.to_skip[m]
        batch_size = 500
        year2id = dataset.time_dict
        
        with torch.no_grad():
            c_begin = 0
            chunk_size = model.sizes[2]
            while c_begin < model.sizes[2]:
                b_begin = 0
                rhs = model.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):                    
                    if queries.shape[1]>4: #time intervals exist
                        these_queries = queries[b_begin:b_begin + batch_size]
                        start_queries = []
                        end_queries = []
                        for triple in these_queries:
                            if triple[3].split('-')[0] == '####':
                                start_idx = -1
                                start = -5000
                            elif triple[3][0] == '-':
                                start=-int(triple[3].split('-')[1].replace('#', '0'))
                            elif triple[3][0] != '-':
                                start = int(triple[3].split('-')[0].replace('#','0'))
                            if triple[4].split('-')[0] == '####':
                                end_idx = -1
                                end = 5000
                            elif triple[4][0] == '-':
                                end =-int(triple[4].split('-')[1].replace('#', '0'))
                            elif triple[4][0] != '-':
                                end = int(triple[4].split('-')[0].replace('#','0'))
                            for key, time_idx in sorted(year2id.items(), key=lambda x:x[1]):
                                if start>=key[0] and start<=key[1]:
                                    start_idx = time_idx
                                if end>=key[0] and end<=key[1]:
                                    end_idx = time_idx


                            if start_idx < 0:
                                start_queries.append([int(triple[0]), int(triple[1])+model.sizes[1]//4, int(triple[2]), end_idx])
                            else:
                                start_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            if end_idx < 0:
                                end_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            else:
                                end_queries.append([int(triple[0]), int(triple[1])+model.sizes[1]//4, int(triple[2]), end_idx])

                        start_queries = torch.from_numpy(np.array(start_queries).astype('int64')).cuda()
                        end_queries = torch.from_numpy(np.array(end_queries).astype('int64')).cuda()

                        q_s = model.get_queries(start_queries)
                        q_e = model.get_queries(end_queries)
                        score = q_s @ rhs + q_e @ rhs
                        target = model.score(start_queries)+model.score(end_queries)
                        
###############  sum up scores of facts from all models ####### 
                        scores[m][b_begin:b_begin+batch_size,:] += score.cpu()
                        targets[m][b_begin:b_begin+batch_size] += target.cpu()
                    else:
                        these_queries = queries[b_begin:b_begin + batch_size]
                        q = model.get_queries(these_queries)
                        """
                        if use_left_queries:
                            lhs_queries = torch.ones(these_queries.size()).long().cuda()
                            lhs_queries[:,1] = (these_queries[:,1]+self.sizes[1]//2)%self.sizes[1]
                            lhs_queries[:,0] = these_queries[:,2]
                            lhs_queries[:,2] = these_queries[:,0]
                            lhs_queries[:,3] = these_queries[:,3]
                            q_lhs = self.get_lhs_queries(lhs_queries)
                            scores = q @ rhs +  q_lhs @ rhs
                            targets = self.score(these_queries) + self.score(lhs_queries)
                        """
                        
                        score = q @ rhs 
                        target = model.score(these_queries)
                        scores[m][b_begin:b_begin+batch_size,:] += score.cpu()
                        targets[m][b_begin:b_begin+batch_size] += target.cpu()
                     
                    b_begin += batch_size
                c_begin += chunk_size
                    
###############  get the final ranks #######           
for m in missing:
    if dataset.interval:## for datasets YAGO11k and wikidata12k, q is numpy.array(str)
        year2id = dataset.time_dict
        examples = test
        q = np.copy(examples)
        if m == 'lhs':
            tmp = np.copy(q[:, 0])
            q[:, 0] = q[:, 2]
            q[:, 2] = tmp
            q[:, 1] = q[:, 1].astype('uint64')+dataset.n_predicates // 2
    
    else:
        examples = torch.from_numpy(test.astype('int64')).cuda()
        q = examples.clone()
        if m == 'lhs':
            tmp = torch.clone(q[:, 0])
            q[:, 0] = q[:, 2]
            q[:, 2] = tmp
            q[:, 1] += dataset.n_predicates // 2
            
    queries = q
    filters = dataset.to_skip[m]
    batch_size = 500
    year2id = dataset.time_dict

    ranks = torch.ones(len(queries))
    with torch.no_grad():
        c_begin = 0
        chunk_size = model.sizes[2]
        while c_begin < model.sizes[2]:            
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                for i, query in enumerate(these_queries):
                    if queries.shape[1]>4:
                        filter_out = filters[int(query[0]), int(query[1]), query[3], query[4]]
                        filter_out += [int(queries[b_begin + i, 2])]                            
                    else:    
                        filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                    if chunk_size < model.sizes[2]:
                        filter_in_chunk = [
                            int(x - c_begin) for x in filter_out
                            if c_begin <= x < c_begin + chunk_size
                        ]
                        scores[m][b_begin+i,torch.LongTensor(filter_in_chunk)] = -1e6
                    else:
                        scores[m][b_begin+i,torch.LongTensor(filter_out)] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores[m][b_begin:b_begin+batch_size,:] >= targets[m][b_begin:b_begin+batch_size]).float(), dim=1
                ).cpu()

                b_begin += batch_size
                
            c_begin += chunk_size
        
        mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
        mean_rank[m] = torch.mean(ranks).item()
        hits_at[m] = torch.FloatTensor((list(map(
            lambda x: torch.mean((ranks <= x).float()).item(),
            at
    ))))    
            

###############  print MRR and Hits #######     
def avg_both(mrs: Dict[str, float], mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    mr = (mrs['lhs'] + mrs['rhs']) / 2.
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MR': mr, 'MRR': m, 'hits@[1,3,10]': h}

results = avg_both(mean_rank, mean_reciprocal_rank, hits_at)
print(results)
filename = ''
for i in range(num):
    filename += str(component[i]['index'])
f = open(os.path.join(PATH, 'result'+filename+'.txt'), 'w+')
f.write("\n model:")
for i in range(num):
    f.write("\n\n")
    f.write(str(component[i]))
f.write("\n\nTEST : ")
f.write(str(results))
f.close()

