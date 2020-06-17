# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
from typing import Dict
import logging
import torch
from torch import optim

from datasets import TemporalDataset
from optimizers import TKBCOptimizer, IKBCOptimizer
from models import ComplEx, TComplEx, TNTComplEx, TGeomE1, TGeomE2
from regularizers import N3, Lambda3
import os

parser = argparse.ArgumentParser(
    description="Temporal ComplEx"
)
parser.add_argument(
    '--dataset', type=str,
    help="Dataset name"
)
models = [
    'ComplEx', 'TComplEx', 'TNTComplEx', 'TGeomE1', 'TGeomE2'
]
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)
parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=100, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)
parser.add_argument(
    '--time_granularity', default=1, type=int, 
    help="Time granularity for time embeddings"
)


args = parser.parse_args()

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
            """
            aggregate metrics for missing lhs and rhs
            :param mrrs: d
            :param hits:
            :return:
            """
            m = (mrrs['lhs'] + mrrs['rhs']) / 2.
            h = (hits['lhs'] + hits['rhs']) / 2.
            return {'MRR': m, 'hits@[1,3,10]': h}

def learn(model=args.model,
          dataset=args.dataset,
          rank=args.rank,
          learning_rate = args.learning_rate,
          batch_size = args.batch_size, 
          emb_reg=args.emb_reg, 
          time_reg=args.time_reg,
          time_granularity=args.time_granularity):

    root = 'results/'+ dataset +'/' + model
    modelname = model
    
    ##restore model parameters and results
    PATH=os.path.join(root,'rank{:.0f}/lr{:.4f}/batch{:.0f}/emb_reg{:.5f}/time_reg{:.5f}'.format(rank,learning_rate,batch_size,emb_reg,time_reg))

    dataset = TemporalDataset(args.dataset)

    sizes = dataset.get_shape()
    model = {
        'ComplEx': ComplEx(sizes, args.rank),
        'TComplEx': TComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
        'TNTComplEx': TNTComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
        'TGeomE1': TGeomE1(sizes, args.rank, no_time_emb=args.no_time_emb, args.time_granularity=time_granularity),
        'TGeomE2': TGeomE2(sizes, args.rank, time_granularity=args.time_granularity)
    }[args.model]
    model = model.cuda()


    opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)

    emb_reg = N3(args.emb_reg)
    time_reg = Lambda3(args.time_reg)
    
    # Results related
    os.makedirs(PATH)
    patience = 0
    mrr_std = 0
    cur_loss = 0
    curve = {'train': [], 'valid': [], 'test': []}

    for epoch in range(args.max_epochs):
        examples = torch.from_numpy(
            dataset.get_train().astype('int64')
        )

        model.train()
        if dataset.has_intervals():
            optimizer = IKBCOptimizer(
                model, emb_reg, time_reg, opt, dataset,
                batch_size=args.batch_size
            )
            optimizer.epoch(examples)

        else:
            optimizer = TKBCOptimizer(
                model, emb_reg, time_reg, opt,
                batch_size=args.batch_size
            )
            optimizer.epoch(examples)

        
        if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
            if dataset.has_intervals():
                valid, test, train = [
                    dataset.eval(model, split, -1 if split != 'train' else 50000)
                    for split in ['valid', 'test', 'train']
                ]
                print("valid: ", valid)
                print("test: ", test)
                print("train: ", train)

            else:
                valid, test, train = [
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['valid', 'test', 'train']
                ]
                print("valid: ", valid['MRR'])
                print("test: ", test['MRR'])
                print("train: ", train['MRR'])

            # Save results

            f = open(os.path.join(PATH, 'result.txt'), 'w+')
            f.write("\n VALID: ")
            f.write(str(valid))
            f.close()
            # early-stop with patience
            mrr_valid = valid['MRR']
            if mrr_valid < mrr_std:
               patience += 1
               if patience >= 3:
                  break
            else:
               patience = 0
               mrr_std = mrr_valid
               torch.save(model.state_dict(), os.path.join(PATH, modelname+'.pkl'))

            curve['valid'].append(valid)
            # curve['test'].append(test)
            curve['train'].append(train)
    
            print("\t TRAIN: ", train)
            print("\t VALID : ", valid)

    model.load_state_dict(torch.load(os.path.join(PATH, modelname+'.pkl')))
    results = avg_both(*dataset.eval(model, 'test', -1))
    print("\n\nTEST : ", results)
    f = open(os.path.join(PATH, 'result.txt'), 'w+')
    f.write("\n\nTEST : ")
    f.write(str(results))
    f.close()

if __name__ == '__main__':


    ## tune parameters here
    for rank in [1000]:
        for lr in [0.1]:
            for batch_size in [1000]:
                for model in ['TGeomE2']:
                    for emb_reg in [0.1,0.11,0.09]:
                        for time_reg in [0.01, 1]:
                            for time_granularity in [1, 2]
                                for dataset in ['ICEWS14', 'ICEWS05-15', 'yago15k']:
                                    learn(  model=model,
                                            dataset=dataset,
                                            rank=rank,
                                            learning_rate = learning_rate,
                                            batch_size = batch_size, 
                                            emb_reg=emb_reg, 
                                            time_reg=time_reg,
                                            time_granularity=time_granularity)
