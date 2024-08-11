from captum.attr import (
    GuidedGradCam, Lime, FeaturePermutation, IntegratedGradients, GradientShap, DeepLift, DeepLiftShap, LayerConductance,)
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np



key = ['urate','phenylalanine','guanine','hypoxanthine','thymine','uracil','trans-aconitate','hydroxyphenylacetate','xanthine','glycerol','homocysteine','dopamine','Orotate','N-Acetylmethionine','Adenine','Selenocystamine','INDOXYL SULFATE','Deoxyribose','THEOBROMINE','TYRAMINE','OXALOACETATE','THIOPURINE_S-METHYLETHER','3-METHOXYTYRAMINE','3-AMINO-4-HYDROXYBENZOATE','3-Methyladenine','Resorcinol_Monoacetate','Ag BG']
def guide_cam_protocol(cls_model,target_layer,device,train_data_loder,visual = True,cls_num = 3):
    save_dir = 'shapley_explain'
    os.makedirs(save_dir,exist_ok=True)
    if not os.path.exists('shapley.pt'):
        cls_model = cls_model.to(device)
        heat_lines = torch.zeros((cls_num, 27))
        heat_lines_counter = torch.zeros(cls_num)
        input_lines = torch.zeros((cls_num, 27))
        cam = GuidedGradCam(cls_model,target_layer)
        shape = GradientShap(cls_model)
        fp = FeaturePermutation(cls_model)
        lime = Lime(cls_model)
        baseline = torch.zeros((1,5433)).to(device)
        for idx,(inputs,target,demo,name) in enumerate(train_data_loder):
            inputs = inputs.float().to(device)
            target = target.long().to(device)[None]
            demo = demo.float().to(device)
            inputs_demo = torch.cat([inputs.reshape(inputs.shape[0],-1),demo],dim=1)
            # print(inputs_demo.shape)
            preds = cls_model(inputs_demo)
            if_accuracy = torch.argmax(preds,dim=1) == target
            print(if_accuracy,'preds',preds,'target',target)
            name = name[0].split(' ')[1]
            if if_accuracy:
                target = target.item()
                for tar in range(3):
                    # print(inputs_demo.shape,baseline.shape)
                    attributions, delta = shape.attribute(inputs_demo, baseline,
                                                          target=target, return_convergence_delta=True)
                    print(attributions.shape)
                    spectra_attributions = attributions[:,:-6].reshape(inputs.shape)
                    demo_attributions = attributions[:,-6:]
                    heat_lines[target] += spectra_attributions[0].mean(0).to('cpu')
                    heat_lines_counter[target] += 1
                    input_lines[target] += inputs[0].mean(0).to('cpu')
                    np.savetxt(os.path.join(save_dir,name+'_GT_'+str(target)+'_tar_'+str(tar)+'.txt'),spectra_attributions[0].cpu())
                    np.savetxt(os.path.join(save_dir,name+'_GT_'+str(target)+'_tar'+str(tar)+'_demo.txt'),demo_attributions[0].cpu())
                # if idx>= 300: break

        heat_lines = heat_lines/heat_lines_counter.unsqueeze(1)
        input_lines = input_lines/heat_lines_counter.unsqueeze(1)

        torch.save(heat_lines,'shapley.pt')
    else:
        heat_lines = torch.load('shapley.pt')
        if visual:
            axis = (np.loadtxt('shift.txt'))
            plt.figure(figsize=(4,6),dpi=200)
            print(heat_lines)
            plt.barh([i for i in range(27)],heat_lines[0].numpy(),color='r',
                    height=1)
            plt.yticks([i for i in range(27)],key,rotation=0)
            plt.subplots_adjust(left=0.75, right=1, top=1, bottom=0.05)
            plt.show()



            plt.figure(figsize=(3, 6), dpi=200)
            plt.barh([i for i in range(27)], heat_lines[1].numpy(), color='r',
                    height=1)
            plt.yticks([i for i in range(27)],key,rotation=0)
            plt.subplots_adjust(left=0.75, right=1, top=1, bottom=0.05)
            plt.show()

            plt.figure(figsize=(3, 6), dpi=200)
            plt.barh([i for i in range(27)], heat_lines[2].numpy(), color='r',
                    height=1)
            plt.yticks([i for i in range(27)],key,rotation=0)
            plt.subplots_adjust(left=0.75, right=1, top=1, bottom=0.05)
            plt.show()


class enc_unsqueeze(nn.Module):
    def __init__(self,enc):
        super(enc_unsqueeze,self).__init__()
        self.enc = enc
    def forward(self,x,y):
        x = self.enc(x,y)
        return x.unsqueeze(0)

class patient_cls(nn.Module):
    def __init__(self,enc,fc):
          super(patient_cls,self).__init__()
          self.enc = enc_unsqueeze(enc)
          self.fc = fc

    def build_patient_graph(self,patient,edge_num):
        # patient: N*L
        # N: number of nodes
        # L: number of features
        N = patient.shape[0]
        index = torch.zeros((2,edge_num))
        affine_value = torch.mm(patient, patient.t())/(torch.norm(patient,dim=1).unsqueeze(1)*torch.norm(patient,dim=1).unsqueeze(0))
        # affine_value = torch.fill_diagonal_(affine_value, 0)
        topk,idx = torch.topk(affine_value.flatten(), k=edge_num)
        index[0] = idx//N
        index[1] = idx%N
        # print(index.shape,topk.shape)
        return index,topk

    def forward(self,x):

        x,demo = x[:,:-6],x[:,-6:]
        x = x.reshape(x.shape[0],201,-1)
        # print('x', x.shape)

        demo = demo.unsqueeze(1)
        feat = torch.zeros((x.shape[0],134)).to(x.device)
        for i in range(x.shape[0]):
            self.patient_edge_num = 500
            self.patient_edge_idx,_ = self.build_patient_graph(x[i],edge_num=self.patient_edge_num)
            feat[i] = torch.cat([self.enc(x[i],self.patient_edge_idx),demo[i]],dim=1)
        x = self.fc(feat)
        # x = torch.softmax(x,dim=0)
        # print(x)
        return x

if __name__ == '__main__':
    from model import emssemble_model
    from dataset import train_loader, test_loader, test_loader_ml, val_loader
    from dataset_interpret import train_loader as loader
    import torch
    import os


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_data, train_demo, train_label = train_loader()
    test_data, test_demo, test_label = test_loader()
    val_data, val_demo, val_label = val_loader()

    train_label = train_label.to(device)
    test_label = test_label.to(device)
    val_label = val_label.to(device)

    sers_groups = torch.cat([train_data, val_data, test_data, ], dim=0).to(device)
    clinic_groups = torch.cat([train_demo, val_demo, test_demo, ], dim=0).to(device)

    train_idx = torch.tensor([i for i in range(len(train_label))]).to(device)
    val_idx = torch.tensor([i for i in range(len(train_label), len(train_label) + len(val_label))]).to(device)
    test_idx = torch.tensor([i for i in range(len(train_label) + len(val_label),
                                              len(train_label) + len(val_label) + len(test_label))]).to(device)

    model_emsse = emssemble_model(group_embed_dim=128, num_classes=3, sers_groups=sers_groups,
                                clinic_groups=clinic_groups, device=device)
    model_name = './model_5.pth'
    if os.path.exists(model_name):
        print('       Existing,load model...')
        model_emsse.load_state_dict(torch.load(model_name))
    model_cls = patient_cls(model_emsse.patient_gcn,model_emsse.lin)
    guide_cam_protocol(model_cls,target_layer = model_cls.enc,device=device,
                 train_data_loder = loader(batch_size=1),
                       visual = True,cls_num = 3)
