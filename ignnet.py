import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import math
import os
import logging
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import copy
import matplotlib.pyplot as plt

def focal_loss(inputs, targets, alpha = 0.25, gamma=2, reduction = 'mean'):
    bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    loss = alpha * (1 - pt) ** gamma * bce_loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

class Block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Block, self).__init__()
        self.fc_1a = nn.Linear(input_dim, output_dim)
        self.fc_1b = nn.Linear(output_dim, output_dim)
        self.fc_2a = nn.Linear(input_dim, output_dim)
        self.fc_2b = nn.Linear(output_dim, output_dim)
        self.concat = nn.Linear(output_dim + output_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, num_features, dim)
        x = torch.cat((self.fc_1b(self.fc_1a(x)),
                       self.fc_2b(self.fc_2a(x))), dim=2)
        x = self.relu(self.concat(x))
        return x

class FNN(nn.Module):
    def __init__(self, num_features):
        super(FNN, self).__init__()
        self.fc_1 = nn.Linear(576, 128) # Input Dim : 64+256+256 = 576
        self.fc_2 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(num_features)
        self.fc_3 = nn.Linear(64, 32)
        self.fc_4 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(num_features)
        self.fc_5 = nn.Linear(16, 8)
        self.fc_6 = nn.Linear(8, 4)
        self.bn3 = nn.BatchNorm1d(num_features)
        self.fc_7 = nn.Linear(4, 2)
        self.fc_8 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.bn1(self.fc_2(self.fc_1(x)))
        x = self.bn2(self.fc_4(self.fc_3(x)))
        x = self.bn3(self.fc_6(self.fc_5(x)))
        x = self.fc_8(self.fc_7(x))
        return x

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.cross_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim)) for _ in range(num_layers)]
        )
        self.cross_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)]
        )

    def forward(self, x0):
        # x0: (batch, input_dim)
        x = x0
        for i in range(self.num_layers):
            dot = torch.sum(x * self.cross_weights[i], dim=1, keepdim=True)  # (batch, 1)
            x = x0 * dot + self.cross_biases[i] + x
        return x

def normalize_adj_matrix(adj):
    D = torch.sum(adj, dim=0)
    D_hat = torch.diag((D)**(-0.5))
    adj_normalized = torch.mm(torch.mm(D_hat, adj), D_hat)
    return adj_normalized

class IGNNet(nn.Module):
    def __init__(self, input_dim, num_features, adj, num_classes, index_to_name, loss):
        super(IGNNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.block1 = Block(64, 64)
        self.block2 = Block(64, 128)
        self.fc4 = nn.Linear(64 + 128, 256)
        self.bn1 = nn.BatchNorm1d(num_features)
        self.block3 = Block(256, 256)
        self.bn2 = nn.BatchNorm1d(num_features)
        self.block4 = Block(256, 512)
        self.bn3 = nn.BatchNorm1d(num_features)
        self.fc7 = nn.Linear(256 + 512, 256)
        self.fnn = FNN(num_features)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.adj = adj
        self.index_to_name = index_to_name
        self.num_classes = num_classes
        
        self.cross_net = CrossNetwork(num_features, num_layers=2)
        self.final_fc = nn.Linear(2 * num_features, 1)
        self.batch_adj = None
        self.loss = loss
        
    def load_batch_adj(self, x_in):
        bs = x_in.shape[0]
        adj_3d = np.zeros((bs, self.adj.shape[0], self.adj.shape[1]), dtype=float)
        for i in range(bs):
            adj_3d[i] = self.adj.cpu()
        adj_train = torch.FloatTensor(adj_3d)
        self.batch_adj = adj_train.to(x_in.device)

    def gnn_forward(self, x_in):
        self.load_batch_adj(x_in)
        x = self.fc1(x_in)  # (batch, num_features, 64)
        x1 = self.relu(torch.bmm(self.batch_adj, x))
        x1 = self.block1(x1)
        x2 = self.relu(torch.bmm(self.batch_adj, x1))
        x2 = self.block2(x2)
        x3 = self.relu(torch.bmm(self.batch_adj, x2))
        x4 = torch.cat((x3, x1), dim=2)
        x4 = self.fc4(x4)
        x4 = self.bn1(x4)
        x5 = self.relu(torch.bmm(self.batch_adj, x4))
        x5 = self.block3(x5)
        x5 = self.bn2(x5)
        x6 = self.relu(torch.bmm(self.batch_adj, x5))
        x6 = self.block4(x6)
        x7 = self.relu(torch.bmm(self.batch_adj, x6))
        x7 = torch.cat((x7, x4), dim=2)
        x7 = self.bn3(self.fc7(x7))
        x = torch.cat((x7, x4, x1), dim=2)
        x = self.fnn(x)
        x = self.sigmoid(x)
        return x

    def forward(self, x_in):
        x_gnn = self.gnn_forward(x_in)
        x_gnn_flat = x_gnn.view(x_gnn.size(0), -1) 
        x_flat = x_in.squeeze(-1)
        x_cross = self.cross_net(x_flat) 
        combined = torch.cat([x_gnn_flat, x_cross], dim=1)
        
        if self.loss == 'BCE':
            combined = torch.cat([x_gnn_flat, x_cross], dim=1)
            out = self.final_fc(combined)
        else:
            out = self.final_fc(combined)
            out = self.sigmoid(out)
        return out

    def predict(self, x_in):
        return self.forward(x_in)

    def get_local_importance(self, x_in):
        x = self.gnn_forward(x_in)
        x = x.view(x.size(0), x.size(1))
        return x.cpu().data.numpy()

    def get_global_importance(self, y):
        feature_global_importances = {}
        weights = self.weights.cpu().detach().numpy()
        for i, name in self.index_to_name.items():
            feature_global_importances[name] = weights[i]
        return feature_global_importances

    def plot_bars(self, normalized_instance, instance, num_f):
        y = self.predict(normalized_instance)
        if self.num_classes > 2:
            y = np.argmax(y[0].cpu().detach().numpy())
        feature_global_importance = self.get_global_importance(y)
        local_importance = self.get_local_importance(normalized_instance).reshape(-1)
        original_values = instance.to_dict()
        names = []
        values = []
        for i, v in enumerate(local_importance):
            name = self.index_to_name[i]
            names.append(name)
            values.append(feature_global_importance[name] * v)
        feature_local_importance = {}
        for i, v in enumerate(values):
            feature_local_importance[self.index_to_name[i]] = v
        feature_names = [f'{name} = {original_values[name]}' for name, val in sorted(feature_local_importance.items(), key=lambda item: abs(item[1]))]
        feature_values = [val for name, val in sorted(feature_local_importance.items(), key=lambda item: abs(item[1]))]
        plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 12, 'font.weight': 'bold'})
        center = 0
        plt.barh(feature_names[-num_f:], feature_values[-num_f:], left=center,
                 color=np.where(np.array(feature_values[-num_f:]) < 0, 'dodgerblue', '#f5054f'))
        for index, v in enumerate(feature_values[-num_f:]):
            if v > 0:
                plt.text(v + center, index, "+{:.2f}".format(v), ha='center')
            else:
                plt.text(v + center, index, "{:.2f}".format(v), ha='left')
        plt.xlabel('Importance')
        plt.rcParams["figure.figsize"] = (8, 8)
        plt.show()

    @property
    def weights(self):
        return torch.mean(torch.stack(list(self.cross_net.cross_weights)), dim=0)

def train_model(input_dim, num_features, adj_matrix, index_to_name,
                train_dataloader, val_dataloader,
                data_name, num_classes, now, loss, num_epochs=300,
                learning_rate=1e-03, normalize_adj=False):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj_matrix = torch.FloatTensor(adj_matrix)
    if normalize_adj:
        adj_matrix = normalize_adj_matrix(adj_matrix)
    
    if loss == 'BCE':
        pos_weight = torch.tensor([2.0], device=device)
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss == 'FOCAL':
        loss_function = focal_loss
    else:
        loss_function = torch.nn.BCELoss()
    
    gnn_model = IGNNet(input_dim, num_features, adj_matrix.to(device), num_classes, index_to_name, loss).to(device)
    optimizer_train = torch.optim.Adam(gnn_model.parameters(), lr=learning_rate)
    
    logger = logging.getLogger(__name__)
    
    best_score = 0.5
    best_epoch = 0
    best_model_wts = copy.deepcopy(gnn_model.state_dict())
    
    for epoch in range(1, num_epochs + 1):
        gnn_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_labels = []
        for i, data in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch}")):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer_train.zero_grad()
            outputs = gnn_model(inputs)
            loss_val = loss_function(outputs.reshape(-1), labels.float())
            loss_val.backward()
            optimizer_train.step()
            
            preds = (outputs.reshape(-1) > 0.5).long()
            train_correct += torch.sum(preds == labels.data).item()
            train_total += len(labels)
            train_loss += loss_val.item()
            train_labels.extend(labels.tolist())
            torch.cuda.empty_cache()
        
        train_accuracy = train_correct / train_total

        gnn_model.eval()
        val_correct = 0
        list_prediction = []
        val_labels = []
        list_prob_pred = []
        for i, data in enumerate(tqdm(val_dataloader, desc=f"Validation Epoch {epoch}")):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = gnn_model(inputs)
            preds = (outputs.reshape(-1) > 0.5).long()
            val_correct += torch.sum(preds == labels.data).item()
            val_labels.extend(labels.tolist())
            list_prediction.extend(preds.tolist())
            list_prob_pred.extend(outputs.tolist())
            torch.cuda.empty_cache()
        
        acc = val_correct / len(val_labels)
        roc = roc_auc_score(val_labels, list_prediction)
        prec = precision_score(val_labels, list_prediction, average='macro')
        recall_metric = recall_score(val_labels, list_prediction, average='macro')
        f_score = f1_score(val_labels, list_prediction, average='macro')
        
        mean_score = (acc + roc + f_score) / 3
        
        logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
                    f"Val Acc = {acc:.4f}, ROC = {roc:.4f}, F1 = {f_score:.4f}, Mean Score = {mean_score:.4f}")
        print('--------------------')
        print(f"Epoch {epoch} - Validation Accuracy: {acc:.2f}")
        print("Train Accuracy: {:.2f}".format(train_accuracy))
        print('--------------------')
        
        if mean_score >= best_score:
            best_score = mean_score
            best_epoch = epoch
            best_model_wts = copy.deepcopy(gnn_model.state_dict())
    
    gnn_model.load_state_dict(best_model_wts)
    logger.info(f"Best model found at epoch {best_epoch} with Mean Score = {best_score:.4f}")
    
    model_dir = './plot/model/'
    optm_dir = './plot/optm/'
    if not os.path.exists(model_dir) :
        os.makedirs(model_dir, exist_ok=True)
        
    if not os.path.exists(optm_dir):
        os.makedirs(optm_dir, exist_ok=True)
    
    mean_str = "{:.2f}".format(best_score)
    model_name = f'test4_{now}_{mean_str}.model'
    optm_name = f'test4_{now}_{mean_str}.optm'
    
    model_path = os.path.join(model_dir, model_name)
    optm_path = os.path.join(optm_dir, optm_name)
    
    torch.save(gnn_model.state_dict(), model_path)
    torch.save(optimizer_train.state_dict(), optm_path)
    
    logger.info(f"Final best model saved: {model_path}")
    
    return gnn_model

def save_importance_plot(model, normalized_instance, instance, num_f, save_path):

    feature_dir = './plot/feature/'
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    save_path = os.path.join(feature_dir, save_path)
    
    y = model.predict(normalized_instance)
    if model.num_classes > 2:
        y = np.argmax(y[0].cpu().detach().numpy())
    
    feature_global_importance = model.get_global_importance(y)
    local_importance = model.get_local_importance(normalized_instance).reshape(-1)
    
    feature_local_importance = {}
    original_values = instance.to_dict()
    
    for i, v in enumerate(local_importance):
        name = model.index_to_name[i]
        combined_importance = feature_global_importance[name] * v
        feature_local_importance[name] = combined_importance
    
    sorted_importances = sorted(feature_local_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    
    top_features = sorted_importances[:num_f]
    
    feature_names = [f"{f_name} = {original_values[f_name]}" for f_name, val in top_features]
    values = [val for f_name, val in top_features]
    
    feature_names.reverse()
    values.reverse()
    
    colors = ['dodgerblue' if v < 0 else '#f5054f' for v in values]
    
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 12, 'font.weight': 'bold'})
    plt.figure(figsize=(8, 6))
    
    plt.barh(feature_names, values, color=colors)
    plt.axvline(x=0, color='black', linewidth=1)
    
    for i, v in enumerate(values):
        if v >= 0:
            plt.text(v + 0.01, i, f"+{v:.2f}", va='center', ha='left')
        else:
            plt.text(v - 0.01, i, f"{v:.2f}", va='center', ha='right')
    
    plt.xlabel("Value")
    plt.title("Feature Importances (SHAP-like) - Top {} Features".format(num_f))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()