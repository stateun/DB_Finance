import pandas as pd
import numpy as np
import scipy.io
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

# VarianceThreshold
from sklearn.feature_selection import VarianceThreshold


# =====================
# Our Data Imbalanced Normal 95% , Abnormal 5%
# =====================

oversampling_save_results = './sampling'

if not os.path.exists(oversampling_save_results):
    os.makedirs(oversampling_save_results)
    

    
class CustomDataset(Dataset) :
    def __init__(self, dataset, data_name, sampling_method, seed, alpha, number = 6, before_plot = False, after_plot = False):
        
        #### Hyperparameter ####
        # alpha : Hyperparameter for VarianceThreshold
        # number : Set the K of k_neighbors
        ########################
        
        print("========== Dataset loading process starting ==========-")
        ori_data = dataset
        save_file_path = f"{oversampling_save_results}/{data_name}"
        
        ###
        TSNE_path = f'./plot/TSNE/{data_name}'
        if not os.path.exists(TSNE_path) :
            os.makedirs(TSNE_path)
        ###
        
        self.data = ori_data.drop(['FraudFound_P'], axis = 1)
        self.label = ori_data['FraudFound_P']
        self.seed = seed
        
        # Feature Selection
        selector = VarianceThreshold(threshold = alpha)
        X_reduced = selector.fit_transform(self.data)
        selected_feature_indices = selector.get_support(indices = True)
        selected_feature_indices = self.data.columns[selected_feature_indices]
        self.data = pd.DataFrame(X_reduced, columns = selected_feature_indices)
        
        print("* Data Shape after Feature Selection : ", self.data.shape)
            
        if before_plot :

            if not os.path.exists(TSNE_path) :
                os.makedirs(TSNE_path)
                
            tsne = TSNE(n_components=2, random_state = self.seed, perplexity = 30)
            tsne_data = tsne.fit_transform(self.data)
            
            print("========== plot the result of TSNE before oversampling ==========")
            plt.scatter(tsne_data[:,0], tsne_data[:,1], alpha = 0.7, c = self.label, cmap = 'viridis')
            plt.colorbar()
            plt.savefig(TSNE_path + '/' + '{}_{}_before.png'.format(sampling_method, seed))
            plt.clf()
            
        
        if sampling_method == 'smote' :
            smote = SMOTE(k_neighbors = max(1, number -1), random_state = self.seed)
            balanced_train_data, balanced_train_labels = smote.fit_resample(self.data, self.label)
            oversampling_data = pd.concat([balanced_train_data, balanced_train_labels], axis = 1)
            oversampling_data.to_csv(save_file_path + '.csv', index = False, encoding='utf-8')
            
        elif sampling_method == 'borderline-smote' :
            borderline_smote = BorderlineSMOTE(k_neighbors = max(1, number - 1), random_state = self.seed)
            balanced_train_data, balanced_train_labels = borderline_smote.fit_resample(self.data, self.label)
            oversampling_data = pd.concat([balanced_train_data, balanced_train_labels], axis = 1)
            oversampling_data.to_csv(save_file_path + '.csv', index = False, encoding='utf-8')
            
        elif sampling_method == 'adasyn' :
            adasyn = ADASYN(random_state = self.seed)
            balanced_train_data, balanced_train_labels = adasyn.fit_resample(self.data, self.label)
            oversampling_data = pd.concat([balanced_train_data, balanced_train_labels], axis = 1)
            oversampling_data.to_csv(save_file_path + '.csv', index = False, encoding='utf-8') 
            
        elif sampling_method == 'over-random' :
            over_random = RandomOverSampler(random_state = self.seed)
            balanced_train_data, balanced_train_labels = over_random.fit_resample(self.data, self.label)
            oversampling_data = pd.concat([balanced_train_data, balanced_train_labels], axis = 1)
            oversampling_data.to_csv(save_file_path + '.csv', index = False, encoding='utf-8')                       
            
        else :
            raise ValueError(f"Upsupported sampling method{sampling_method}. Please Check the your code.")
        
        if sampling_method != 'none' and after_plot == True :
            print("Data Shape after oversampling :", oversampling_data.shape)
            print("====================")
            
            if not before_plot :
                tsne = TSNE(n_components=2, random_state = self.seed, perplexity = 30)
                tsne_data = tsne.fit_transform(balanced_train_data)
                
                print("========== plot the result of TSNE after oversampling ==========")
                plt.scatter(tsne_data[:,0], tsne_data[:,1], alpha = 0.7, c = balanced_train_labels, cmap = 'viridis')
                plt.colorbar()
                plt.savefig(TSNE_path + '/' + '{}_{}_after.png'.format(sampling_method, seed))
                plt.clf()
                
            else :
                tsne_data = tsne.fit_transform(balanced_train_data)
                
                print("========== plot the result of TSNE after oversampling ==========")
                plt.scatter(tsne_data[:,0], tsne_data[:,1], alpha = 0.7, s = 20, c = balanced_train_labels, cmap = 'viridis')
                plt.colorbar()
                plt.savefig(TSNE_path + '/' + '{}_{}_after.png'.format(sampling_method, seed))
                plt.clf()
            
        print("========== Feature Selection & Oversampling Process Complete. ==========")
        
        self.train_data = balanced_train_data
        self.train_y = balanced_train_labels
        
        self.train_dataset = pd.concat([balanced_train_data, balanced_train_labels], axis = 1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]