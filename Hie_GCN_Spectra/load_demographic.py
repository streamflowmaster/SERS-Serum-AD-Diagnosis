import pandas
import numpy as np
import torch
data = pandas.read_excel('demographic information.xlsx')
data = data.to_numpy()
print('idx',data[:,-1])
print('age',data[:,1])
print('Gender (1:male; 2:female)',data[:,2])
print('MMSE',data[:,3])
idx = np.where(data[:,4] == '/')
data[idx,4] = 3
print('MOCA',data[:,4])
print('Diagnosis (1AD, 2MCI, 3HC)',data[:,5])
print('Education level (文盲:1, 小学:2, 初中: 3, 高中：4，大学：5)',data[:,6])

idx = np.where(data[:,7] == '/')
data[idx,7] = 1
print('Aβ (1:positive; 0:negative))',data[:,7])


for i in range(data.shape[0]):
    name = 'demographic/['+str(data[i,5])+'] '+str(data[i,-1]).zfill(3)+' nobase.pt'
    print(name)
    obj = torch.tensor([
        int(data[i,1]),
        int(data[i,2]),
        int(data[i,3]),
        int(data[i,4]),
        int(data[i,6]),
        int(data[i,7]),
    ])
    torch.save(obj,name)