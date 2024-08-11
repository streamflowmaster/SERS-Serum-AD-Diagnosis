from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot
from model import emssemble_model
import torch.nn as nn
from dataset import train_loader,test_loader,test_loader_ml,val_loader
import torch
import os
from evaluation import evaluate

def draw_curve(label,predict, visual = True):

    fpr = dict()
    tpr = dict()
    if type(label) == torch.Tensor:
        label = label.detach().cpu().numpy()
    if type(predict) == torch.Tensor:
        predict = predict.detach().cpu().numpy()
    print(predict.shape,label.shape)

    print('AUC:',roc_auc_score(label,predict, multi_class='ovr'))
    fpr["micro"], tpr["micro"], _  = roc_curve(label.ravel(), predict.ravel())
    if visual:
        print('fpr',fpr["micro"],
              'tpr',tpr["micro"])
        # matplotlib.pyplot.plot()
        matplotlib.pyplot.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})',
                 color='deeppink', linestyle='-', linewidth=2)

        matplotlib.pyplot.show()
    return fpr["micro"], tpr["micro"]



def main():
    # 1. load data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_data,train_demo,train_label = train_loader()
    val_data,val_demo,val_label = val_loader()
    test_data,test_demo,test_label = test_loader()

    sers_groups = torch.cat([train_data,val_data,test_data],dim=0).to(device)
    clinic_groups = torch.cat([train_demo,val_demo,test_demo],dim=0).to(device)
    train_label = train_label.to(device)
    test_label = test_label.to(device)

    # 2. build model
    model = emssemble_model(group_embed_dim=128,num_classes=3,sers_groups=sers_groups,
                            clinic_groups=clinic_groups,device=device)
    # print(model)
    print('model parameters: ', sum(param.numel() for param in model.parameters()))
    print('trainable parameters: ', sum(param.numel() for param in model.parameters() if param.requires_grad))
    print('group edge num: ', model.group_edge_num)
    print('patient edge num: ', model.patient_edge_num)

    # 3. train model
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=0.99)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    if os.path.exists( './model_5.pth'):
        print('       Existing,load model...')
        model.load_state_dict(torch.load('./model_5.pth'))

    eva = evaluate(cls_num=3)
    model = model.eval()
    optimizer.zero_grad()
    pred, pred_paient = model(sers_groups,clinic_groups)
    # test_loss = criterion(pred[-len(test_label):],test_label)
    # print(pred,test_label)
    eva.calculation(test_label.cpu(),
                    torch.argmax(pred[-len(test_label):],
                                 dim=1).cpu())
    eva.eval()
    return pred[-len(test_label):],test_label

if __name__ == '__main__':
    pred,label = main()
    pred = torch.softmax(pred,dim=1)
    label = torch.nn.functional.one_hot(label, num_classes=3)
    print(label.shape,pred.shape)
    draw_curve(label,pred)