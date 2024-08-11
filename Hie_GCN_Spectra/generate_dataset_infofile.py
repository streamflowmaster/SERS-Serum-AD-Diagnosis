'''
dataset_info_file:
dict={
'train':[[file_path,label],...]
'val':[[file_path,label],...]
'test':[[file_path,label],...]
}
'''
import numpy as np
import os



def read_dataset_info(info_path='dataset_info_file.npy'):
    info = np.load(info_path,allow_pickle = True).item()
    # print(info)
    print(len(info['train']))
    print(len(info['test']))
    print(len(info['val']))


def generate_seg_PN_info(hsi_path,label,tvt_ratio=[0.8,0.1,0.1],map_dict={'train':[],'val':[],'test':[]}):
    H_list = os.listdir(hsi_path)
    H_len = len(H_list)
    print(H_len)
    tvt_ratio = np.array(tvt_ratio)
    tvt_ratio = tvt_ratio/sum(tvt_ratio)
    N_sampler = np.argsort(np.random.uniform(low = 0, high=1,size=H_len))
    train_set = N_sampler[:int(H_len*tvt_ratio[0])]
    val_set = N_sampler[int(H_len*tvt_ratio[0]):int(H_len*tvt_ratio[0])+int(H_len * tvt_ratio[1])]
    test_set = N_sampler[int(H_len*tvt_ratio[0])+int(H_len * tvt_ratio[1]):]
    for file_idx in train_set:
        n_path = hsi_path + H_list[file_idx]
        l_path = label
        map_dict['train'].append([n_path,l_path])
    for file_idx in val_set:
        n_path = hsi_path + H_list[file_idx]
        l_path = label
        map_dict['val'].append([n_path,l_path])
    for file_idx in test_set:
        n_path = hsi_path + H_list[file_idx]
        l_path = label
        map_dict['test'].append([n_path,l_path])
    # print(map_dict)
    return map_dict

def generate_seg_info_file(L_path, N_path, P_path,tvt_ratio=[0.8,0.1,0.1],save_name='AD_dataseg_3.npy'):
    map_dict = {'train': [], 'val': [], 'test': []}
    map_dict = generate_seg_PN_info(hsi_path = L_path, label = 0, map_dict = map_dict,tvt_ratio=tvt_ratio)
    map_dict = generate_seg_PN_info(hsi_path = N_path,label = 1, map_dict = map_dict,tvt_ratio=tvt_ratio)
    map_dict = generate_seg_PN_info(hsi_path = P_path,label = 2, map_dict = map_dict,tvt_ratio=tvt_ratio)
    print(len(map_dict['train'])+len(map_dict['test'])+len(map_dict['val']))
    print(len(map_dict['train']))
    print(len(map_dict['test']))
    np.save(save_name,map_dict)

generate_seg_info_file(L_path = '1/',
                       N_path = '2/',
                       P_path = '3/',)

# generate_seg_info_file(L_path = '1wb/',
#                        N_path = '2wb/',
#                        P_path = '3wb/',save_name='AD_dataseg_3_wb.npy')

