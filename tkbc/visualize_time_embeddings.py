import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from datasets import TemporalDataset
from models import ComplEx, TComplEx, TNTComplEx, TGeomE1, TGeomE2, TGeomE3

import os
import pickle
import pkg_resources
from pathlib import Path

device = 'cuda'

data_dir = 'ICEWS14'
dataset = TemporalDataset(data_dir)

component={}
component['model']='TGeomE2'
component['rank']=2000
component['lr']=0.1
component['batch']=1000	
component['emb_reg']=0.0075
component['time_reg']=0.01
component['time_granularity']=1
component['epoch_pretrain']=50
component['fact_count']=1

root = 'results/'+ data_dir +'/' + component['model']
PATH=os.path.join(root,'rank{:.0f}/lr{:.4f}/batch{:.0f}/time_granularity{:02d}/emb_reg{:.5f}/time_reg{:.5f}/epoch_pretrain{:.5f}/fact_count{:d}'.format(component['rank'],
                    component['lr'],component['batch'],component['time_granularity'], component['emb_reg'], component['time_reg'], component['epoch_pretrain'], component['fact_count']))


sizes = dataset.get_shape()
model = {
    'ComplEx': ComplEx(sizes, component['rank']),
    'TComplEx': TComplEx(sizes, component['rank'], no_time_emb=False),
    'TNTComplEx': TNTComplEx(sizes, component['rank'], no_time_emb=False),
    'TGeomE1': TGeomE1(sizes, component['rank'], no_time_emb=False, time_granularity=component['time_granularity']),
    'TGeomE2': TGeomE2(sizes, component['rank'], no_time_emb=False, time_granularity=component['time_granularity']),
    'TGeomE3': TGeomE3(sizes, component['rank'], no_time_emb=False, time_granularity=component['time_granularity'])
}[component['model']]

model.load_state_dict(torch.load(os.path.join(PATH, component['model'] +'.pkl')))
model.to(device)
print('model{:s}'.format(component['model'])+' loaded')


pca = PCA(n_components=2)
time_embeddings = model.embeddings[2].weight.data.cpu().numpy()

ids = range(365)
time_embeddings_pca = pca.fit_transform(time_embeddings)
dates = np.array('2014-01-01', dtype=np.datetime64) + np.arange(365)
dates = pd.to_datetime(dates).date

fig = plt.figure(figsize=(6, 5))
fig.suptitle('PCA analysis for time embeddings: {:s} on {:s}'.format(component['model'], data_dir))
plt.scatter(time_embeddings_pca[:,0], time_embeddings_pca[:,1])

for i, time_embedding_pca, date in zip(ids, time_embeddings_pca, dates):
    if date.day == 1 or date.day == 2 or date.day == 3:
        plt.annotate(date, (time_embeddings_pca[i,0], time_embeddings_pca[i,1]))


# plt.legend()
# plt.show()
plt.savefig("time_embs_visualize.png")
