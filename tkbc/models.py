# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import math
import torch
from torch import nn
import numpy as np


class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    @abstractmethod
    def forward_over_time(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries,
            filters, year2id = {},
            batch_size: int = 1000, chunk_size: int = -1, use_left_queries:bool = False
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
					if queries.shape[1]>4: #map time stamps in Wikidata12k and YAGO11k to time index
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
                                start_queries.append([int(triple[0]), int(triple[1])+self.sizes[1]//4, int(triple[2]), end_idx])
                            else:
                                start_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                                
                            if end_idx < 0:
                                end_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            else:
                                end_queries.append([int(triple[0]), int(triple[1])+self.sizes[1]//4, int(triple[2]), end_idx])
                            
                        start_queries = torch.from_numpy(np.array(start_queries).astype('int64')).cuda()
                        end_queries = torch.from_numpy(np.array(end_queries).astype('int64')).cuda()
                        
                        q_s = self.get_queries(start_queries)
                        q_e = self.get_queries(end_queries)
                        scores = q_s @ rhs + q_e @ rhs
                        targets = self.score(start_queries)+self.score(end_queries)
			
			
					else:
						these_queries = queries[b_begin:b_begin + batch_size]
						q = self.get_queries(these_queries)

						if use_left_queries:
							lhs_queries = torch.ones(these_queries.size()).long().cuda()
							lhs_queries[:,1] = (these_queries[:,1]+self.sizes[1]//2)%self.sizes[1]
							lhs_queries[:,0] = these_queries[:,2]
							lhs_queries[:,2] = these_queries[:,0]
							lhs_queries[:,3] = these_queries[:,3]
							q_lhs = self.get_lhs_queries(lhs_queries)

							scores = q @ rhs +  q_lhs @ rhs
							targets = self.score(these_queries) + self.score(lhs_queries)
						else:
							scores = q @ rhs 
							targets = self.score(these_queries)

                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
					
                    for i, query in enumerate(these_queries):
                        if queries.shape[1]>4: #for datasets YAGO11k and Wikidata12k
                            filter_out = filters[int(query[0]), int(query[1]), query[3], query[4]]
                            filter_out += [int(queries[b_begin + i, 2])]                            
                        else:    
                            filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                            filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

    def get_auc(
            self, queries: torch.Tensor, batch_size: int = 1000
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, begin, end)
        :param batch_size: maximum number of queries processed at once
        :return:
        """
        all_scores, all_truth = [], []
        all_ts_ids = None
        with torch.no_grad():
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                scores = self.forward_over_time(these_queries)
                all_scores.append(scores.cpu().numpy())
                if all_ts_ids is None:
                    all_ts_ids = torch.arange(0, scores.shape[1]).cuda()[None, :]
                assert not torch.any(torch.isinf(scores) + torch.isnan(scores)), "inf or nan scores"
                truth = (all_ts_ids <= these_queries[:, 4][:, None]) * (all_ts_ids >= these_queries[:, 3][:, None])
                all_truth.append(truth.cpu().numpy())
                b_begin += batch_size

        return np.concatenate(all_truth), np.concatenate(all_scores)

    def get_time_ranking(
            self, queries: torch.Tensor, filters: List[List[int]], chunk_size: int = -1
    ):
        """
        Returns filtered ranking for a batch of queries ordered by timestamp.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: ordered filters
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            q = self.get_queries(queries)
            targets = self.score(queries)
            while c_begin < self.sizes[2]:
                rhs = self.get_rhs(c_begin, chunk_size)
                scores = q @ rhs
                # set filtered and true scores to -1e6 to be ignored
                # take care that scores are chunked
                for i, (query, filter) in enumerate(zip(queries, filters)):
                    filter_out = filter + [query[2].item()]
                    if chunk_size < self.sizes[2]:
                        filter_in_chunk = [
                            int(x - c_begin) for x in filter_out
                            if c_begin <= x < c_begin + chunk_size
                        ]
                        max_to_filter = max(filter_in_chunk + [-1])
                        assert max_to_filter < scores.shape[1], f"fuck {scores.shape[1]} {max_to_filter}"
                        scores[i, filter_in_chunk] = -1e6
                    else:
                        scores[i, filter_out] = -1e6
                ranks += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()

                c_begin += chunk_size
        return ranks


class ComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    @staticmethod
    def has_time():
        return False

    def forward_over_time(self, x):
        raise NotImplementedError("no.")

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]
        return (
                       (lhs[0] * rel[0] - lhs[1] * rel[1]) @ right[0].transpose(0, 1) +
                       (lhs[0] * rel[1] + lhs[1] * rel[0]) @ right[1].transpose(0, 1)
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), None

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)


class TComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.no_time_emb = no_time_emb

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
             lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1]) * rhs[0] +
            (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
             lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                       (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
                       (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        return torch.cat([
            lhs[0] * rel[0] * time[0] - lhs[1] * rel[1] * time[0] -
            lhs[1] * rel[0] * time[1] - lhs[0] * rel[1] * time[1],
            lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
            lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]
        ], 1)


class TNTComplEx(TKBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2
    ):
        super(TNTComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3], sizes[1]]  # last embedding modules contains no_time embeddings
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        self.embeddings[3].weight.data *= init_size

        self.no_time_emb = no_time_emb

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        return torch.sum(
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0] +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rel_no_time = self.embeddings[3](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        rrt = rt[0] - rt[3], rt[1] + rt[2]
        full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]

        regularizer = (
           math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
           torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
           torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
           math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )
        return ((
               (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
               (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
            ), regularizer,
               self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        )

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        rel_no_time = self.embeddings[3](x[:, 1])
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        score_time = (
            (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
             lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
            (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
             lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )
        base = torch.sum(
            (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
             lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
            (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
             lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
            dim=1, keepdim=True
        )
        return score_time + base

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rel_no_time = self.embeddings[3](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

        return torch.cat([
            lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
            lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
        ], 1)


class TGeomE1(TKBCModel):
    """1st-grade Temporal Knowledge Graph Embeddings using Geometric Algebra

        :::     Scoring function: <h, r, t_conjugate, T>
        :::     1-grade multivector = scalar + Imaginary * e_1

    """
    
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, time_granularity: int = 1
):
        super(TGeomE1, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        # self.embeddings[0].weight.data[:,self.rank:] *= 0
        # self.embeddings[0].weight.data[:,self.rank:] *= 0
        

        self.no_time_emb = no_time_emb
        self.time_granularity = time_granularity

    @staticmethod
    def has_time():
        return True

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3] // self.time_granularity) 

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] * time[0] + lhs[1] * rel[1] * time[0] +
             lhs[1] * rel[0] * time[1] + lhs[0] * rel[1] * time[1]) * rhs[0] -
            (lhs[1] * rel[0] * time[0] + lhs[0] * rel[1] * time[0] +
             lhs[0] * rel[0] * time[1] + lhs[1] * rel[1] * time[1]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3] // self.time_granularity) 

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] + rt[3], rt[1] + rt[2]

        return (
                       (lhs[0] * full_rel[0] + lhs[1] * full_rel[1]) @ right[0].t() -
                       (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        return (
                (lhs[0] * rel[0] * rhs[0] + lhs[1] * rel[1] * rhs[0] -
                 lhs[1] * rel[0] * rhs[1] - lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
                (lhs[1] * rel[0] * rhs[0] + lhs[0] * rel[1] * rhs[0] -
                 lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3] // self.time_granularity) 
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]
        return torch.cat([
            lhs[0] * rel[0] * time[0] + lhs[1] * rel[1] * time[0] +
            lhs[1] * rel[0] * time[1] + lhs[0] * rel[1] * time[1],
            -lhs[1] * rel[0] * time[0] - lhs[0] * rel[1] * time[0] -
            lhs[0] * rel[0] * time[1] - lhs[1] * rel[1] * time[1]
        ], 1)



class TGeomE2(TKBCModel):
    """2nd-grade Temporal Knowledge Graph Embeddings using Geometric Algebra

        :::     Scoring function: <h, r, t_conjugate, T>
        :::     2-grade multivector = scalar + Imaginary * e_1 + Imaginary * e_2 + Imaginary * e_3 + Imaginary * e_12

    """
    def __init__(
            self, sizes: Tuple[int, int, int, int], rank: int,
            no_time_emb=False, init_size: float = 1e-2, time_granularity: int = 1,
	    pre_train: bool = True
):
        super(TGeomE2, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 4 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]] # without no_time_emb
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        
        self.pre_train = pre_train
	
        if self.pre_train:
      #      self.embeddings[0].weight.data[:,self.rank:self.rank*3] *= 0
            self.embeddings[1].weight.data[:,self.rank:self.rank*3] *= 0
      #      self.embeddings[2].weight.data[:,self.rank:self.rank*3] *= 0
        

        self.no_time_emb = no_time_emb

        self.time_granularity = time_granularity

    @staticmethod
    def has_time():
        return True
	

    def score(self, x):

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3] // self.time_granularity) 
        lhs = lhs[:, :self.rank], lhs[:, self.rank:self.rank*2], lhs[:, self.rank*2:self.rank*3], lhs[:, self.rank*3:]
        rel = rel[:, :self.rank], rel[:, self.rank:self.rank*2], rel[:, self.rank*2:self.rank*3], rel[:, self.rank*3:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:self.rank*2], rhs[:, self.rank*2:self.rank*3], rhs[:, self.rank*3:]
        time = time[:, :self.rank], time[:, self.rank:self.rank*2], time[:, self.rank*2:self.rank*3], time[:, self.rank*3:]

        # compute <h, r, T, t_conj> ==> 4**3
        # h * r
        # A =   lhs[0]*rel[0]+ lhs[1]*rel[1]+ lhs[2]*rel[2]- lhs[3]*rel[3] # scalar
        # B =   lhs[0]*rel[1]+ lhs[1]*rel[0]- lhs[2]*rel[3]+ lhs[3]*rel[2] # e1
        # C =   lhs[0]*rel[2]+ lhs[2]*rel[0]+ lhs[1]*rel[3]- lhs[3]*rel[1]  # e2
        # D =   lhs[1]*rel[2]- lhs[2]*rel[1]+ lhs[0]*rel[3]+ lhs[3]*rel[0] # e1e2
        # # (h*r) * time, note that we first change the +- sign for easier dot product later
        # W =   A * time[0]+ B * time[1]+ C * time[2]- D * time[3] # scalar
        # X = - A * time[1]- B * time[0]+ C * time[3]- D * time[2] # e1
        # Y = - A * time[2]- C * time[0]- B * time[3]+ D * time[1] # e2
        # Z =   B * time[2]- C * time[1]+ A * time[3]+ D * time[0] # e1e2
	
	
	    ## compute <h, r, T, t_conj> ==> 4**3
	    ## full_rel = r * time
        A =   rel[0]*time[0]+ rel[1]*time[1]+ rel[2]*time[2]- rel[3]*time[3] # scalar
        B =   rel[0]*time[1]+ rel[1]*time[0]- rel[2]*time[3]+ rel[3]*time[2] # e1
        C =   rel[0]*time[2]+ rel[2]*time[0]+ rel[1]*time[3]- rel[3]*time[1]  # e2
        D =   rel[1]*time[2]- rel[2]*time[1]+ rel[0]*time[3]+ rel[3]*time[0] # e1e2
	    
        full_rel = A,B,C,D
	    ## h * full_rel, note that we do not change +- sign here, thus we need do that later
        W =   lhs[0]*full_rel[0]+ lhs[1]*full_rel[1]+ lhs[2]*full_rel[2]- lhs[3]*full_rel[3] # scalar
        X =   lhs[0]*full_rel[1]+ lhs[1]*full_rel[0]- lhs[2]*full_rel[3]+ lhs[3]*full_rel[2] # e1
        Y =   lhs[0]*full_rel[2]+ lhs[2]*full_rel[0]+ lhs[1]*full_rel[3]- lhs[3]*full_rel[1]  # e2
        Z =   lhs[1]*full_rel[2]- lhs[2]*full_rel[1]+ lhs[0]*full_rel[3]+ lhs[3]*full_rel[0] # e1e2
	
	
	
        # return torch.sum(W*rhs[0] + X * rhs[1] + Y * rhs[2] + Z * rhs[3], 1, keepdim=True)
        #  return h * full_rel * t_conj, note that here the signs before X and Y are -
        return torch.sum(W*rhs[0] - X * rhs[1] - Y * rhs[2] + Z * rhs[3], 1, keepdim=True)
	
	
	
    def pretrain(self, x):
        
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3] // self.time_granularity)

        lhs = lhs[:, :self.rank], lhs[:, self.rank*3:]
        rel = rel[:, :self.rank], rel[:, self.rank*3:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank*3:]
        time = time[:, :self.rank], time[:, self.rank*3:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank*3:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
                       (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ to_score[0].t() +
                       (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ to_score[1].t()
               ),(
                       (rhs[0] * full_rel[0] + rhs[1] * full_rel[1]) @ to_score[0].t() +
                       (rhs[1] * full_rel[0] - rhs[0] * full_rel[1]) @ to_score[1].t()
	       ),(
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else torch.cat((self.embeddings[2].weight[:,:self.rank],self.embeddings[2].weight[:,
	       self.rank*3:]),dim=1)


    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3] // self.time_granularity) 

        lhs = lhs[:, :self.rank], lhs[:, self.rank:self.rank*2], lhs[:, self.rank*2:self.rank*3], lhs[:, self.rank*3:]
        rel = rel[:, :self.rank], rel[:, self.rank:self.rank*2], rel[:, self.rank*2:self.rank*3], rel[:, self.rank*3:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:self.rank*2], rhs[:, self.rank*2:self.rank*3], rhs[:, self.rank*3:]
        time = time[:, :self.rank], time[:, self.rank:self.rank*2], time[:, self.rank*2:self.rank*3], time[:, self.rank*3:]


        # compute <h, r, T, t_conj> ==> 4**4 / 2 
        # h * r
        # A =   lhs[0]*rel[0]+ lhs[1]*rel[1]+ lhs[2]*rel[2]- lhs[3]*rel[3] # scalar
        # B =   lhs[0]*rel[1]+ lhs[1]*rel[0]- lhs[2]*rel[3]+ lhs[3]*rel[2] # e1
        # C =   lhs[0]*rel[2]+ lhs[2]*rel[0]+ lhs[1]*rel[3]- lhs[3]*rel[1]  # e2
        # D =   lhs[1]*rel[2]- lhs[2]*rel[1]+ lhs[0]*rel[3]+ lhs[3]*rel[0] # e1e2
        # # (h*r) * time, note that we first change the +- sign for easier dot product later
        # W =   A * time[0]+ B * time[1]+ C * time[2]- D * time[3] # scalar
        # X = - A * time[1]- B * time[0]+ C * time[3]- D * time[2] # e1
        # Y = - A * time[2]- C * time[0]- B * time[3]+ D * time[1] # e2
        # Z =   B * time[2]- C * time[1]+ A * time[3]+ D * time[0] # e1e2
	
	
        # compute <h, r, T, t_conj> ==> 4**3
        # full_rel = r * time
        A =   rel[0]*time[0]+ rel[1]*time[1]+ rel[2]*time[2]- rel[3]*time[3] # scalar
        B =   rel[0]*time[1]+ rel[1]*time[0]- rel[2]*time[3]+ rel[3]*time[2] # e1
        C =   rel[0]*time[2]+ rel[2]*time[0]+ rel[1]*time[3]- rel[3]*time[1]  # e2
        D =   rel[1]*time[2]- rel[2]*time[1]+ rel[0]*time[3]+ rel[3]*time[0] # e1e2

        full_rel = A,B,C,D
        
        # h * full_rel, note that we do not change +- sign here, thus we need do that later
        W =   lhs[0]*full_rel[0]+ lhs[1]*full_rel[1]+ lhs[2]*full_rel[2]- lhs[3]*full_rel[3] # scalar
        X =   lhs[0]*full_rel[1]+ lhs[1]*full_rel[0]- lhs[2]*full_rel[3]+ lhs[3]*full_rel[2] # e1
        Y =   lhs[0]*full_rel[2]+ lhs[2]*full_rel[0]+ lhs[1]*full_rel[3]- lhs[3]*full_rel[1]  # e2
        Z =   lhs[1]*full_rel[2]- lhs[2]*full_rel[1]+ lhs[0]*full_rel[3]+ lhs[3]*full_rel[0] # e1e2
	
        W1 =  full_rel[0]*rhs[0]- full_rel[1]*rhs[1]- full_rel[2]*rhs[2]+ full_rel[3]*rhs[3]
        X1 =  full_rel[1]*rhs[0]- full_rel[0]*rhs[1]- full_rel[3]*rhs[2]+ full_rel[2]*rhs[3]
        Y1 =  full_rel[2]*rhs[0]+ full_rel[3]*rhs[1]- full_rel[0]*rhs[2]- full_rel[1]*rhs[3]
        Z1 =- full_rel[3]*rhs[0]- full_rel[2]*rhs[1]+ full_rel[1]*rhs[2]+ full_rel[0]*rhs[3]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:self.rank*2], to_score[:, self.rank*2:self.rank*3], to_score[:, self.rank*3:]

        # return (
        #             W @ to_score[0].transpose(0, 1) +
        #             X @ to_score[1].transpose(0, 1) +
        #             Y @ to_score[2].transpose(0, 1) +
        #             Z @ to_score[3].transpose(0, 1)
        #        ), (
        #             torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2+ lhs[2] ** 2+ lhs[3] ** 2),
        #             torch.sqrt(rel[0] ** 2 + rel[1] ** 2+ rel[2] ** 2+ rel[3] ** 2),
		#     torch.sqrt(time[0] ** 2 + time[1] ** 2+ time[2] ** 2+ time[3] ** 2),
        #             torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2+ rhs[2] ** 2+ rhs[3] ** 2)
        #        ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight
        
        # note that the signs before X and Y are -
        return (
                    W @ to_score[0].transpose(0, 1) -
                    X @ to_score[1].transpose(0, 1) -
                    Y @ to_score[2].transpose(0, 1) +
                    Z @ to_score[3].transpose(0, 1)
               ),(
                    W1 @ to_score[0].transpose(0, 1) +
                    X1 @ to_score[1].transpose(0, 1) +
                    Y1 @ to_score[2].transpose(0, 1) +
                    Z1 @ to_score[3].transpose(0, 1)
               ),(
                    torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2+ lhs[2] ** 2+ lhs[3] ** 2),
                    torch.sqrt(full_rel[0] ** 2 + full_rel[1] ** 2+ full_rel[2] ** 2+ full_rel[3] ** 2),
                    torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2+ rhs[2] ** 2+ rhs[3] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight

    def forward_over_time(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2].weight

        lhs = lhs[:, :self.rank], lhs[:, self.rank:self.rank*2], lhs[:, self.rank*2:self.rank*3], lhs[:, self.rank*3:]
        rel = rel[:, :self.rank], rel[:, self.rank:self.rank*2], rel[:, self.rank*2:self.rank*3], rel[:, self.rank*3:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:self.rank*2], rhs[:, self.rank*2:self.rank*3], rhs[:, self.rank*3:]
        time = time[:, :self.rank], time[:, self.rank:self.rank*2], time[:, self.rank*2:self.rank*3], time[:, self.rank*3:]

        # compute <h, r, t_conj, ?> ==> 4**4 / 2 
        # h * r
        A =   lhs[0]*rel[0]+ lhs[1]*rel[1]+ lhs[2]*rel[2]- lhs[3]*rel[3] # scalar
        B = - lhs[0]*rel[1]- lhs[1]*rel[0]+ lhs[2]*rel[3]- lhs[3]*rel[2] # e1
        C = - lhs[0]*rel[2]- lhs[2]*rel[0]- lhs[1]*rel[3]+ lhs[3]*rel[1]  # e2
        D =   lhs[1]*rel[2]- lhs[2]*rel[1]+ lhs[0]*rel[3]+ lhs[3]*rel[0] # e1e2
        # (h*r) * t_conj
        W =   A * rhs[0] - B * rhs[1] - C * rhs[2] + D * rhs[3] # scalar
        X = - A * rhs[1] + B * rhs[0] + C * rhs[3] - D * rhs[2] # e1   +  B * rhs[0]
        Y = - A * rhs[2] + C * rhs[0] - B * rhs[3] + D * rhs[1] # e2
        Z = - B * rhs[2] + C * rhs[1] - A * rhs[3] + D * rhs[0] # e1e2 + C * rhs[1]

        to_time = self.embeddings[2].weight
        to_time = to_time[:, :self.rank], to_time[:, self.rank:self.rank*2], to_time[:, self.rank*2:self.rank*3], to_time[:, self.rank*3:]
        return (
                W @ time[0].transpose(0, 1) +
                X @ time[1].transpose(0, 1) +
                Y @ time[2].transpose(0, 1) +
                Z @ time[3].transpose(0, 1)
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3] // self.time_granularity) 
        lhs = lhs[:, :self.rank], lhs[:, self.rank:self.rank*2], lhs[:, self.rank*2:self.rank*3], lhs[:, self.rank*3:]
        rel = rel[:, :self.rank], rel[:, self.rank:self.rank*2], rel[:, self.rank*2:self.rank*3], rel[:, self.rank*3:]
        time = time[:, :self.rank], time[:, self.rank:self.rank*2], time[:, self.rank*2:self.rank*3], time[:, self.rank*3:]

        # compute <h, r, T, t_conj> ==> 4**4 / 2 
        # h * r
        A =   lhs[0]*rel[0]+ lhs[1]*rel[1]+ lhs[2]*rel[2]- lhs[3]*rel[3] # scalar
        B =   lhs[0]*rel[1]+ lhs[1]*rel[0]- lhs[2]*rel[3]+ lhs[3]*rel[2] # e1
        C =   lhs[0]*rel[2]+ lhs[2]*rel[0]+ lhs[1]*rel[3]- lhs[3]*rel[1]  # e2
        D =   lhs[1]*rel[2]- lhs[2]*rel[1]+ lhs[0]*rel[3]+ lhs[3]*rel[0] # e1e2
        # (h*r) * time, note that we first change the +- sign for easier dot product later
        W =   A * time[0]+ B * time[1]+ C * time[2]- D * time[3] # scalar
        X = - A * time[1]- B * time[0]+ C * time[3]- D * time[2] # e1
        Y = - A * time[2]- C * time[0]- B * time[3]+ D * time[1] # e2
        Z =   B * time[2]- C * time[1]+ A * time[3]+ D * time[0] # e1e2

        return torch.cat([W,X,Y,Z], 1)

    def get_lhs_queries(self, queries: torch.Tensor):
        rhs = self.embeddings[0](queries[:, 2])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3] // self.time_granularity)
	
        rel = rel[:, :self.rank], rel[:, self.rank:self.rank*2], rel[:, self.rank*2:self.rank*3], rel[:, self.rank*3:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:self.rank*2], rhs[:, self.rank*2:self.rank*3], rhs[:, self.rank*3:]
        time = time[:, :self.rank], time[:, self.rank:self.rank*2], time[:, self.rank*2:self.rank*3], time[:, self.rank*3:]
	
        # compute <h, r, T, t_conj> ==> 4**3
        # full_rel = r * time
        A =   rel[0]*time[0]+ rel[1]*time[1]+ rel[2]*time[2]- rel[3]*time[3] # scalar
        B =   rel[0]*time[1]+ rel[1]*time[0]- rel[2]*time[3]+ rel[3]*time[2] # e1
        C =   rel[0]*time[2]+ rel[2]*time[0]+ rel[1]*time[3]- rel[3]*time[1]  # e2
        D =   rel[1]*time[2]- rel[2]*time[1]+ rel[0]*time[3]+ rel[3]*time[0] # e1e2

        full_rel = A,B,C,D
        
        # h * full_rel
	
        W1 =  full_rel[0]*rhs[0]- full_rel[1]*rhs[1]- full_rel[2]*rhs[2]+ full_rel[3]*rhs[3]
        X1 =  full_rel[1]*rhs[0]- full_rel[0]*rhs[1]- full_rel[3]*rhs[2]+ full_rel[2]*rhs[3]
        Y1 =  full_rel[2]*rhs[0]+ full_rel[3]*rhs[1]- full_rel[0]*rhs[2]- full_rel[1]*rhs[3]
        Z1 =- full_rel[3]*rhs[0]- full_rel[2]*rhs[1]+ full_rel[1]*rhs[2]+ full_rel[0]*rhs[3]
        return torch.cat([W1,X1,Y1,Z1], 1)
