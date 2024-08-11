from model import emssemble_model
import torch.nn as nn
from dataset import train_loader,test_loader,test_loader_ml,val_loader
import torch
import os
from evaluation import evaluate

def main():
    # 1. load data
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    train_data,train_demo,train_label = train_loader('AD_dataseg_3_resave.npy')
    test_data,test_demo,test_label = test_loader('AD_dataseg_3_resave.npy')
    val_data,val_demo,val_label = val_loader('AD_dataseg_3_resave.npy')

    train_label = train_label.to(device)
    test_label = test_label.to(device)
    val_label = val_label.to(device)

    sers_groups = torch.cat([train_data,val_data,test_data,],dim=0).to(device)
    clinic_groups = torch.cat([train_demo,val_demo,test_demo,],dim=0).to(device)

    train_idx = torch.tensor([i for i in range(len(train_label))]).to(device)
    val_idx = torch.tensor([i for i in range(len(train_label),len(train_label)+len(val_label))]).to(device)
    test_idx = torch.tensor(    [i for i in range(len(train_label) + len(val_label),
                      len(train_label) + len(val_label) + len(test_label))]).to(device)

    # 2. build model
    model = emssemble_model(group_embed_dim=128,num_classes=3,sers_groups=sers_groups,
                            clinic_groups=clinic_groups,device=device)
    # print(model)
    model_name = './model_6.pth'
    print('model parameters: ', sum(param.numel() for param in model.parameters()))
    print('trainable parameters: ', sum(param.numel() for param in model.parameters() if param.requires_grad))
    print('group edge num: ', model.group_edge_num)
    print('patient edge num: ', model.patient_edge_num)

    # 3. train model
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=0.99)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    if os.path.exists(model_name):
        print('       Existing,load model...')
        model.load_state_dict(torch.load(model_name))


    eva = evaluate(cls_num=3)
    model = model.eval()
    pred,pred_paient = model(sers_groups,clinic_groups)
    val_loss = criterion(pred[val_idx],val_label)
    val_acc = (torch.argmax(pred[val_idx],
                            dim=1) == val_label).sum().item() / len(val_label)
    print('val loss: %.3f' % (val_loss.item()))
    print('val acc: %.3f' % (val_acc))
    eva.calculation(val_label.cpu(),
                    torch.argmax(pred[val_idx],
                    dim=1).cpu())
    eva.eval()
    test_loss = criterion(pred[test_idx],test_label)
    test_acc = (torch.argmax(pred[test_idx],
                             dim=1) == test_label).sum().item() / len(test_label)
    print(test_label)
    print(torch.argmax(pred[test_idx],dim=1))
    print('test loss: %.3f' % (test_loss.item()))
    print('test acc: %.3f' % (test_acc))
    eva = evaluate(cls_num=3)
    eva.calculation(test_label.cpu(),
                    torch.argmax(pred[-len(test_label):],
                    dim=1).cpu())
    eva.eval()
    eva.show()

    test_loss = criterion(pred_paient[test_idx],test_label)
    print(test_label)
    print(torch.argmax(pred_paient[test_idx],dim=1))
    test_acc = (torch.argmax(pred_paient[test_idx],
                             dim=1) == test_label).sum().item() / len(test_label)
    print('test loss_patient: %.3f' % (test_loss.item()))
    print('test acc_patient: %.3f' % (test_acc))
    eva = evaluate(cls_num=3)
    eva.calculation(test_label.cpu(),
                    torch.argmax(pred_paient[test_idx],
                    dim=1).cpu())
    eva.eval()
    eva.show()



if __name__ == '__main__':
    main()