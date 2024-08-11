from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
from dataset import train_loader,test_loader,test_loader_ml
class PatientGCN(torch.nn.Module):
    def __init__(self, node_features,
                 hidden_channels,
                 graph_embed_dim,device):
        super(PatientGCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(node_features, hidden_channels).to(device)
        self.conv2 = GCNConv(hidden_channels, hidden_channels).to(device)
        self.conv3 = GCNConv(hidden_channels, hidden_channels).to(device)
        self.conv4 = GCNConv(hidden_channels, hidden_channels).to(device)
        self.conv5 = GCNConv(hidden_channels, hidden_channels).to(device)
        self.lin = Linear(hidden_channels,graph_embed_dim).to(device)
        self.device = device
    def forward(self, x, edge_index):
        edge_index = edge_index.int().to(self.device)
        # print(x.shape)
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        # 2. Readout layer
        # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # print(x.shape)
        x = x.max(0)[0]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

class GroupGCN(torch.nn.Module):
    def __init__(self,group_embed_dim,num_classes,device):
        super(GroupGCN, self).__init__()
        self.conv1 = GCNConv(group_embed_dim, 64).to(device)  # 输入=节点特征维度，16是中间隐藏神经元个数
        self.conv2 = GCNConv(64, 32).to(device)
        self.conv3 = GCNConv(32, 16).to(device)
        self.conv4 = GCNConv(16, num_classes).to(device)
        self.device = device
    def forward(self, x, edge_index):
        # print(x.shape,edge_index.shape)
        edge_index = edge_index.int().to(self.device)
        x = x.to(self.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)
    def forward_1(self,x, edge_index):
        # print(x.shape,edge_index.shape)
        edge_index = edge_index.int().to(self.device)
        x = x.to(self.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        return x

    def forward_2(self,x, edge_index):
        # print(x.shape,edge_index.shape)
        edge_index = edge_index.int().to(self.device)
        x = x.to(self.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

    def forward_3(self,x, edge_index):
        # print(x.shape,edge_index.shape)
        edge_index = edge_index.int().to(self.device)
        x = x.to(self.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        return x

    def forward_4(self,x, edge_index):
        # print(x.shape,edge_index.shape)
        edge_index = edge_index.int().to(self.device)
        x = x.to(self.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        return x

class emssemble_model(torch.nn.Module):
    def __init__(self, group_embed_dim, num_classes,sers_groups,clinic_groups,device):
        super(emssemble_model, self).__init__()
        self.group_embed_dim = group_embed_dim
        self.patient_gcn = PatientGCN(node_features=sers_groups.shape[-1],
                                      hidden_channels=128,
                                      graph_embed_dim=group_embed_dim,
                                      device = device)
        self.group_gcn = GroupGCN(group_embed_dim=group_embed_dim+clinic_groups.shape[-1],
                                  num_classes=num_classes,
                                  device = device)
        # self.lin = Linear(num_classes, num_classes)
        self.patient_edge_num = 500
        self.group_edge_num = 5000
        self.patient_edge_idx,_ = self.build_patient_graphs(sers_groups,edge_num=self.patient_edge_num)
        self.group_edge_idx,_ = self.build_group_graph(clinic_groups,edge_num=self.group_edge_num)
        self.group_edge_idx = self.group_edge_idx.int().to(device)
        self.patient_edge_idx = self.patient_edge_idx.int().to(device)
        self.lin = Linear(134,num_classes)
        self.device = device

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

    def build_patient_graphs(self,patients,edge_num):
        # patients: B*N*L
        indices = []
        values = []
        for i in range(patients.shape[0]):
            index,value = self.build_patient_graph(patients[i],edge_num)
            indices.append(index)
            values.append(value)
        indices = torch.stack(indices,dim=0)
        values = torch.stack(values,dim=0)
        # print(indices.shape,values.shape)
        return indices,values


    def build_group_graph(self,groups,edge_num):
        # groups: B*N
        # B: batch size
        # N: number of features
        B,N = groups.shape
        index = torch.zeros((2, edge_num))
        # select the clinical features as the edge building envidence
        # print(groups)
        groups = groups[:,2]
        # print(groups)
        Mat = groups.unsqueeze(0).repeat(B,1)
        INF = torch.ones_like(Mat)*torch.max(Mat).item()
        Mat = Mat - Mat.t()
        # Mat = torch.fill_diagonal_(Mat,torch.max(Mat)[0])
        Mat = torch.triu(Mat)
        INF = torch.tril(INF)
        Mat = torch.abs(Mat)+INF
        # print(Mat)
        value,idx = torch.topk(-Mat.flatten(),k=edge_num)
        index[0] = idx // B
        index[1] = idx % B
        index  = index.int()
        # print(index)
        return index,value

    def forward_1(self,x,demographic):
        demographic = demographic.float().to(self.device)
        x = x.float().to(self.device)
        embed_feat = torch.zeros((x.shape[0], self.group_embed_dim), device=self.device)
        for i in range(x.shape[0]):
            embed_feat[i] = self.patient_gcn(x[i], self.patient_edge_idx[i])
        print(embed_feat.shape,demographic.shape)
        embed_feat = torch.cat([embed_feat, demographic], dim=1)
        embed_feat = self.group_gcn.forward_1(embed_feat, self.group_edge_idx)
        # embed_feat = self.lin(embed_feat)
        return embed_feat

    def forward_2(self,x,demographic):
        demographic = demographic.float().to(self.device)
        x = x.float().to(self.device)
        embed_feat = torch.zeros((x.shape[0], self.group_embed_dim), device=self.device)
        for i in range(x.shape[0]):
            embed_feat[i] = self.patient_gcn(x[i], self.patient_edge_idx[i])
        print(embed_feat.shape,demographic.shape)
        embed_feat = torch.cat([embed_feat, demographic], dim=1)
        embed_feat = self.group_gcn.forward_2(embed_feat, self.group_edge_idx)
        # embed_feat = self.lin(embed_feat)
        return embed_feat

    def forward_3(self,x,demographic):
        demographic = demographic.float().to(self.device)
        x = x.float().to(self.device)
        embed_feat = torch.zeros((x.shape[0], self.group_embed_dim), device=self.device)
        for i in range(x.shape[0]):
            embed_feat[i] = self.patient_gcn(x[i], self.patient_edge_idx[i])
        print(embed_feat.shape,demographic.shape)
        embed_feat = torch.cat([embed_feat, demographic], dim=1)
        embed_feat = self.group_gcn.forward_3(embed_feat, self.group_edge_idx)
        # embed_feat = self.lin(embed_feat)
        return embed_feat

    def forward_4(self,x,demographic):
        demographic = demographic.float().to(self.device)
        x = x.float().to(self.device)
        embed_feat = torch.zeros((x.shape[0], self.group_embed_dim), device=self.device)
        for i in range(x.shape[0]):
            embed_feat[i] = self.patient_gcn(x[i], self.patient_edge_idx[i])
        print(embed_feat.shape,demographic.shape)
        embed_feat = torch.cat([embed_feat, demographic], dim=1)
        embed_feat = self.group_gcn.forward_4(embed_feat, self.group_edge_idx)
        # embed_feat = self.lin(embed_feat)
        return embed_feat

    def forward(self,x,demographic):
        demographic = demographic.float().to(self.device)
        x = x.float().to(self.device)
        embed_feat = torch.zeros((x.shape[0],self.group_embed_dim),device=self.device)
        for i in range(x.shape[0]):
            embed_feat[i] = self.patient_gcn(x[i], self.patient_edge_idx[i])
        # print(embed_feat.shape,demographic.shape)
        embed_feat = torch.cat([embed_feat,demographic],dim=1)
        feat_init = embed_feat
        embed_feat = self.group_gcn(embed_feat, self.group_edge_idx)
        # embed_feat = self.lin(embed_feat)
        return embed_feat, F.softmax(self.lin(feat_init),dim=1)


if __name__ == '__main__':
    train_data,train_demo,train_label = train_loader()
    test_data,test_demo,test_label = test_loader()
    sers_groups = torch.cat([train_data,test_data],dim=0)
    clinic_groups = torch.cat([train_demo,test_demo],dim=0)

    model = emssemble_model(group_embed_dim=64,num_classes=3,
                            sers_groups=sers_groups,clinic_groups=clinic_groups,)

    # model = PatientGCN(hidden_channels=64, node_features=1, graph_embed_dim=64)
    print(model(sers_groups).shape)
    print(model)