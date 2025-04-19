import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCN(torch.nn.Module):
    def __init__(self, num_nodes, num_channels, num_classes, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = GCNConv(num_nodes, num_channels)
        self.conv2 = GCNConv(num_channels, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x


class GAT(torch.nn.Module):
    def __init__(self, num_nodes, num_channels, num_classes, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = GATConv(num_nodes, num_channels)
        self.conv2 = GATConv(num_channels, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x


def train(model, data, train_mask, optimizer):
    model.train()
    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


def valid(model, data, valid_mask):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_weight)
    pred = out.argmax(dim=1)
    correct = pred[valid_mask] == data.y[valid_mask]
    acc = int(correct.sum()) / int(valid_mask.sum())

    return acc
