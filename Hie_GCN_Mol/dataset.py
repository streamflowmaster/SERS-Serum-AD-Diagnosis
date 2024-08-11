import torch
import torch.nn as nn
import torch.utils.data as tud
import numpy as np
import os
class dataset(tud.Dataset):
    def __init__(self,dataseg_file,train_or_test,demographic_pth = 'demographic'):
        self.seg_dict = np.load(dataseg_file,allow_pickle=True).item()
        print(self.seg_dict)
        for i in range(len(self.seg_dict['test'])):
            print(self.seg_dict['test'][i][0]+'\n')
        self.filelist = self.seg_dict[train_or_test]
        self.argument_ratio = 2
        self.argument_mixer_num = 4
        self.demographic_pth = demographic_pth
        self.load_data(self.seg_dict[train_or_test])
        # if train_or_test == 'train': self.argumentation()

    def load_data(self,filelist):
        self.data = []
        self.label = []
        self.demographic = []
        for file in filelist:
            data = torch.load(file[0])
            label = file[1]
            self.data.append(data)
            self.label.append(label)
            pth = file[0].split('/')[1]
            self.demographic.append(torch.load(
                os.path.join(self.demographic_pth,pth)))

        self.data = torch.stack(self.data)
        self.label = torch.tensor(self.label)
        self.demographic = torch.stack(self.demographic)
        self.dataset_len = self.data.shape[0]
        # print(self.demographic.shape)

        print(self.label)

    def load_data_(self,filelist):
        datas = []
        labels = []
        demographic = []
        for file in filelist:
            data = torch.load(file[0])
            label = file[1]
            datas.append(data)
            labels.append(label)
            pth = file[0].split('/')[1]
            demographic.append(torch.load(
                os.path.join(self.demographic_pth, pth)))

        datas = torch.stack(datas)
        labels = torch.tensor(labels)
        demographic = torch.stack(demographic)

        return datas,labels,demographic

    def argumentation(self):
        test_data, test_label,demo = self.load_data_(self.seg_dict['test'])
        mix_data = torch.cat([self.data, test_data], dim=0)
        mix_label = torch.cat([self.label, test_label], dim=0)
        mix_demo = torch.cat([self.demographic,demo],dim=0)

        B,N,C = mix_data.shape
        mix_data = mix_data.reshape(B*N,C).float()
        mix_label = mix_label.repeat(1,201).reshape(B*N).float()
        mix_demo = mix_demo.repeat(1,201).reshape(B*N,-1).float()
        # mix_data = self.data.reshape(self.dataset_len*201,-1).float()
        # mix_label = self.label.repeat(1,201).reshape(self.dataset_len*201).float()
        # mix_demo = self.demographic.repeat(1,201).reshape(self.dataset_len*201,-1).float()

        self.argument_len = self.argument_ratio*self.dataset_len*201
        self.argument_data = torch.zeros((self.argument_len,self.data.shape[-1]))
        self.argument_label = torch.zeros(self.argument_len)
        self.argument_demo = torch.zeros((self.argument_len,self.demographic[0].shape[0]))


        for idx in range(self.argument_len):
            # select mixer spectra
            seeds = torch.randint(0,mix_data.shape[0]-1,(self.argument_mixer_num,))
            # print(torch.mode(self.argument_label[seeds]))
            # decide the pseudo label of the mixed spectra
            # print(self.label[seeds])
            pseudo_label,pseudo_idx = torch.mode(mix_label[seeds])
            # print(pseudo_label,pseudo_idx)
            self.argument_label[idx] = pseudo_label.long()
            # decide the mixing ratio
            amp = torch.abs(torch.rand(self.argument_mixer_num))
            amp[pseudo_idx] += 10
            amp = amp/amp.sum()
            amp = amp.unsqueeze(0).float()
            # print(amp,self.data[seeds,:].shape)
            # mix the spectra
            self.argument_data[idx,:] = torch.mm(amp,mix_data[seeds,:])
            self.argument_demo[idx,:] = mix_demo[pseudo_idx,:]
            # print(self.argument_demo.shape)

        self.argument_data = self.argument_data.reshape(self.argument_ratio*self.dataset_len,201,-1)
        self.argument_label = self.argument_label.reshape(self.argument_ratio*self.dataset_len,201)
        print(self.argument_demo.shape)

        self.argument_demo = self.argument_demo.reshape(-1,201,6)
        print(self.argument_demo.shape)
        self.argument_label = torch.mode(self.argument_label,dim=1)[0]
        self.argument_len = self.argument_ratio*self.dataset_len
        self.argument_demo = torch.mode(self.argument_demo,dim=1)[0]

        # print(self.argument_data.shape)
        # print(self.demographic.shape,self.argument_demo.shape)
        self.data = torch.cat([self.data,self.argument_data],dim=0)
        self.label = torch.cat([self.label,self.argument_label],dim=0).long()
        self.demographic = torch.cat([self.demographic,self.argument_demo],dim=0)
        self.dataset_len = self.dataset_len + self.argument_len
        # print(self.argument_demo.shape,self.demographic.shape)
        # print(self.data.shape,self.label.shape,self.demographic.shape)
    def __getitem__(self, idx):
        return self.data[idx],self.label[idx],self.demographic[idx]

    def __len__(self):
        return self.dataset_len

def train_loader(dataseg_file='AD_dataseg_0.npy',batch_size=1):
    train_dataset = dataset(dataseg_file,'train')
    return train_dataset.data,train_dataset.demographic,train_dataset.label,

def test_loader(dataseg_file='AD_dataseg_0.npy',batch_size=1):
    test_dataset = dataset(dataseg_file,'test')
    return test_dataset.data,test_dataset.demographic,test_dataset.label

def val_loader(dataseg_file='AD_dataseg_0.npy',batch_size=1):
    val_dataset = dataset(dataseg_file,'val')
    return val_dataset.data,val_dataset.demographic,val_dataset.label

def train_loader_ml(dataseg_file='AD_dataseg_0.npy'):
    train_dataset = dataset(dataseg_file,'train')
    X,Y= train_dataset.data,train_dataset.label
    return X,Y

def test_loader_ml(dataseg_file='AD_dataseg_0.npy'):
    test_dataset = dataset(dataseg_file,'test')
    X,Y= test_dataset.data,test_dataset.label
    return X,Y

if __name__ == '__main__':
    train_loader = train_loader()
    test_loader = test_loader()
    print(train_loader[0].shape,train_loader[1].shape,train_loader[2].shape)
    print(test_loader[0].shape,test_loader[1].shape,test_loader[2].shape)
  
