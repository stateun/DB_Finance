########## Load Package ##########

# IGNNet Module ( Custom Module )
from ignnet4 import train_model, save_importance_plot
import data_preprocess as dp
import dataset_load as dl

# Graph Model
import networkx as nx
from collections import defaultdict

# Interpretability Model
from data_preprocess import BlackBoxWrapper
import shap

# Environment Variables
import os
os.environ["OMP_NUM_THREADS"] = '4'
os.environ["OMP_THREAD_LIMIT"] = '4'
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["PAPERLESS_AVX2_AVAILABLE"] = "false"
os.environ["OCR_THREADS"] = '4'

# Torch
import torch
from torch.utils.data import DataLoader, Dataset

# Basic Modules
import argparse
from datetime import datetime
import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm
import pickle
import copy
import logging

# Suppress specific sklearn warnings
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

# Sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

####################################################################################

def _parse_args():
    # Basic Hyper Parameter
    parser = argparse.ArgumentParser(description="Runner for IGNnet.")
    parser.add_argument('--data_name', default='revised_adult', type=str,
                        choices=['fraud_data'],
                        help='Dataset')
    parser.add_argument('--file_path', default='./data',
                        type=str, help='Path to the dataset folder')
    parser.add_argument('--output', default='./results',
                        help='Path of the output file')
    parser.add_argument('--gpu', default='1', type=str, help='GPU number')
    parser.add_argument('--seed', type=int, default=25, help='Random seed')
    
    # Training Settings
    parser.add_argument('-e', '--epochs', default=250, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--loss', type=str, default=None, help = 'What loss do you want to use ?')
    
    # Save/Load Model
    parser.add_argument('--save', default=None, type=str,
                        help='Filename to save the trained synthesizer')
    parser.add_argument('--load', default=None, type=str,
                        help='Filename to load a trained synthesizer')
    
    # Evaluation Parameters
    parser.add_argument('--sampling', default=None, type=str,
                        help="What sampling do you want to use?")
    parser.add_argument('--lr', default=0.01, type=int,
                        help="Learning Rate?")
    return parser.parse_known_args()

############## Evaluate Classifier (LR, DT, RF, MLP, SVM, XGBoost) ##############
def evaluate_baseline_models(train_data, train_label, test_data, test_label, seed=25):
    baseline_results = []
    models = {
        "LogisticRegression": LogisticRegression(random_state=seed, max_iter=500),
        "DecisionTree": tree.DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(random_state=seed),
        "MLP": MLPClassifier(random_state=seed, max_iter=300),
        "SVM": svm.SVC(probability=True, random_state=seed),
        "XGBoost": xgb.XGBClassifier(random_state=seed, eval_metric='logloss')
    }
    for model_name, model in models.items():
        model.fit(train_data, train_label)
        if hasattr(model, "predict_proba"):
            prob_pred = model.predict_proba(test_data)[:, 1]
        else:
            prob_pred = model.decision_function(test_data)
        preds = model.predict(test_data)
        
        auc = roc_auc_score(test_label, prob_pred) * 100
        acc = np.mean(preds == test_label) * 100
        prec = precision_score(test_label, preds, average="macro") * 100
        rec = recall_score(test_label, preds, average="macro") * 100
        f1 = f1_score(test_label, preds, average="macro") * 100
        
        best_params = str(model.get_params())
        baseline_results.append({
            "model": model_name,
            "AUC": round(auc, 2),
            "ACC": round(acc, 2),
            "Precision": round(prec, 2),
            "Recall": round(rec, 2),
            "F1_score": round(f1, 2),
            "best_params": best_params
        })
        
        print(f"Model: {model_name}")
        print(f"AUC: {auc:.2f}, Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1-score: {f1:.2f}")
        print("=" * 20)

    return baseline_results

############## XAI : Interpretability_Evaluation , SHAP ##############
def run_interpretability_evaluation(gnn_model , test_tensor, adj_matrix, index_to_name, device, data_name, sampling):
    top_n = 10

    ## Plot the SHAP
    instance_input = test_tensor[10][0].reshape(1, adj_matrix.shape[0], 1).to(device)
    
    interpre_plot = f"./plot/Interpretability/{sampling}/"
    
    if not os.path.exists(interpre_plot):
        os.makedirs(interpre_plot)
    
    importances = gnn_model.weights.cpu().detach().reshape(-1).numpy()
    feature_global_importance = {}
    for i, v in enumerate(importances):
        feature_global_importance[index_to_name[i]] = v
        
    background = torch.zeros_like(instance_input).to(device)
    explainer = shap.GradientExplainer(gnn_model, background)
    shap_values = explainer.shap_values(instance_input)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_vals = shap_values.reshape(-1)

    feature_names = [index_to_name[i] for i in range(len(shap_vals))]
    abs_shap_vals = np.abs(shap_vals)
    top_indices = np.argsort(abs_shap_vals)[::-1][:10]
    top_indices_for_plot = top_indices[::-1]
    top_feature_names = [feature_names[i] for i in top_indices_for_plot]
    top_shap_vals = shap_vals[top_indices_for_plot]

    plt.figure(figsize=(10, 6))
    plt.barh(top_feature_names, top_shap_vals, color='skyblue')
    plt.xlabel("SHAP Value")
    plt.title("SHAP Feature Attributions (GradientExplainer) - Top 10 Features")
    shap_plot = f"./plot/Interpretability/{sampling}/"
    
    if not os.path.exists(shap_plot):
        os.makedirs(shap_plot)
    
    shap_name = f"{data_name}_shap_gradient_top10_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    shap_plot_filename = os.path.join(shap_plot, shap_name)
    
    plt.savefig(shap_plot_filename, bbox_inches='tight')
    plt.close()

    print(f"[XAI] SHAP GradientExplainer top10 plot saved: {shap_plot_filename}")

############## Runner ##############
def run_experiment(args, now):
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = f"cuda:{args.gpu}"
    else:
        print("Please check your environment for a GPU.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    full_path = os.path.join(args.file_path, args.data_name) + '.csv'
    load_data = pd.read_csv(full_path).astype(int)
    encoded_data = load_data.drop(['FraudFound_P'], axis=1)
    label_data = load_data['FraudFound_P']

    train_data, left_out, train_label, y_left_out = train_test_split(
        encoded_data, label_data, test_size=0.2, random_state=args.seed
    )
    test_data, dev_data, test_label, dev_label = train_test_split(
        left_out, y_left_out, test_size=0.5, random_state=args.seed
    )

    train_set = pd.concat([train_data, train_label], axis=1)
    custom_data = dl.CustomDataset(
        dataset=train_set,
        data_name=args.data_name,
        sampling_method=args.sampling,
        seed=args.seed,
        alpha=args.alpha
    )
    print("Oversampled dataset shape: ", custom_data.train_dataset.shape)
    train_data = custom_data.train_dataset.drop(['FraudFound_P'], axis=1)
    train_label = custom_data.train_dataset['FraudFound_P']
    train_column = train_data.columns
    test_data = test_data[train_column]
    dev_data = dev_data[train_column]

    graph_default = './plot/graph'
    if not os.path.exists(graph_default):
        os.makedirs(graph_default)
    graph_name = f"{args.sampling}_{now}.png"
    graph_path = os.path.join(graph_default, graph_name)
    
    adj_matrix, index_to_name, name_to_index = dp.compute_adjacency_matrix(
        data=train_data, self_loop_weight=20, threshold=0.2
    )
    
    G = nx.from_numpy_array(adj_matrix)
    mapping = {i: index_to_name[i] for i in range(len(index_to_name))}
    G = nx.relabel_nodes(G, mapping)
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=args.seed)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title("Graph Representation of Tabular Data")
    plt.axis('off')
    plt.savefig(graph_path, bbox_inches='tight')
    
    train_tensor = dp.transform_to_tensors(train_data, train_label, adj_matrix)
    test_tensor = dp.transform_to_tensors(test_data, test_label, adj_matrix)
    dev_tensor = dp.transform_to_tensors(dev_data, dev_label, adj_matrix)
    train_dataset = dp.Data(data=train_tensor)
    val_dataset = dp.Data(data=dev_tensor)
    test_dataset = dp.Data(data=test_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    num_features = train_data.shape[1] 
    input_dim = 1

    gnn_model = train_model(input_dim, num_features, adj_matrix, index_to_name,
                              train_dataloader, val_dataloader,
                              data_name=args.data_name,
                              now = now,
                              loss = args.loss,
                              num_classes=2,
                              num_epochs=args.epochs,
                              learning_rate=args.lr,
                              normalize_adj=False)
    
    print('========== Start testing IGNnet ==========') 
    gnn_model.eval()
    list_prediction = []
    list_prob_pred = []
    y_test_pred = []
    for i, data in tqdm(list(enumerate(test_dataloader)), desc="Testing IGNnet"):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = gnn_model(inputs)
        list_prob_pred.extend(outputs.tolist())
        preds = (outputs.reshape(-1) > 0.5) * 1
        list_prediction.extend(preds.tolist())
        y_test_pred.extend(labels.tolist())
        torch.cuda.empty_cache()
    
    auc = roc_auc_score(y_test_pred, list_prob_pred) * 100
    prec = precision_score(y_test_pred, list_prediction, average='macro') * 100
    recall_metric = recall_score(y_test_pred, list_prediction, average='macro') * 100
    f_score = f1_score(y_test_pred, list_prediction, average='macro') * 100
    acc = np.mean(np.array(list_prediction) == np.array(y_test_pred)) * 100
    print('========== End testing IGNnet ==========') 
    
    print("=========== IGNnet Test Metrics: =========== ")
    print("AUC: {:.2f}, ACC: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}".format(auc, acc, prec, recall_metric, f_score))
    print(" ============================================ ")
    best_params = f"alpha:{args.alpha}, sampling:{args.sampling}, batch_size:{args.batch_size}, epochs:{args.epochs}, lr:{args.lr}"
    deep_results = {
        "dataset": args.data_name,
        "model": "IGNnet",
        "AUC": round(auc, 2),
        "ACC": round(acc, 2),
        "Precision": round(prec, 2),
        "Recall": round(recall_metric, 2),
        "F1_score": round(f_score, 2),
        "best_params": best_params
    }
    
    baseline_results = evaluate_baseline_models(train_data, train_label, test_data, test_label, seed=args.seed)
    
    run_interpretability_evaluation(gnn_model, test_tensor, adj_matrix, index_to_name, device, args.data_name, 
                                    sampling = args.sampling)
    
    importance_dir = f"./plot/ImportanceExtracted/{args.sampling}/"
    if not os.path.exists(importance_dir):
        os.makedirs(importance_dir)
    importance_filename = f"{args.data_name}_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    importance_save_path = os.path.join(importance_dir, importance_filename)
    save_importance_plot(gnn_model, test_tensor[0][0].reshape(1, adj_matrix.shape[0], 1).to(device), test_data.iloc[0], 10, importance_save_path)
    print(f"[XAI] Importance plot saved: {importance_save_path}")
    
    all_results = [deep_results] + baseline_results
    return all_results

def main():
    args, _ = _parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    log_file = os.path.join(args.output, 'test1.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    logger = logging.getLogger(__name__)
    
    results_csv_path = os.path.join(args.output, "test4_results.csv")
    if not os.path.exists(results_csv_path):
        pd.DataFrame(columns=["dataset", "model", "AUC", "ACC", "Precision", "Recall", "F1_score", "best_params"]).to_csv(
            results_csv_path, index=False
        )
    
    for dataset in ['fraud_data']:
        for batch_size in [64]:
            for epoch in [100]:
                for sampling in ['smote']:
                    for alpha in [0.01, 0.03, 0.05, 0.1]:
                        for lr in [1e-2]:
                            for loss in ['Original']:
                                now = datetime.now().strftime("%H-%M-%S")
                                args.alpha = alpha
                                args.sampling = sampling
                                args.data_name = dataset
                                args.batch_size = batch_size
                                args.epochs = epoch
                                args.lr = lr
                                args.loss = loss
                                
                                logger.info("Running experiment with params:\n"
                                            f"* dataset = {dataset}\n"
                                            f"* batch_size = {batch_size}\n"
                                            f"* epochs = {epoch}\n"
                                            f"* sampling = {sampling}\n"
                                            f"* alpha = {alpha}\n"
                                            f"* lr = {lr}\n"
                                            f"* loss = {loss}")
                                
                                print(f"Running experiment with params :\n"
                                      "============================\n",
                                      f"* alpha={alpha}\n", 
                                      f"* sampling={sampling}\n", 
                                      f"* batch_size={batch_size}\n", 
                                      f"* epochs={epoch}\n", 
                                      f"* lr={lr}\n",
                                      f"* loss={loss}\n"
                                      "============================")
                                
                                exp_results = run_experiment(args, now)
                                
                                results_df = pd.DataFrame(exp_results, columns=["dataset", "model", "AUC", "ACC", "Precision", "Recall", "F1_score", "best_params"])
                                results_df.to_csv(results_csv_path, mode='a', header=False, index=False)
                                logger.info(f"Results appended to {results_csv_path}")

if __name__ == '__main__':
    main()
