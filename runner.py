########## Load Package ##########

# IGNNet Module ( Custom Module )
from ignnet import train_model
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

####################################################################################

def _parse_args():
    
    # Basic Hyper Parameter
    parser = argparse.ArgumentParser(description="Runner for IGNnet.")
    parser.add_argument('--data_name', default='revised_adult', type=str,
                        choices=['fraud_data'],
                        help='Dataset')
    parser.add_argument('--file_path', default='/data/home/stateun/data',
                        type=str, help='Path to the dataset folder')
    parser.add_argument('--output', default='/data/home/stateun/results',
                        help='Path of the output file')
    parser.add_argument('--gpu', default='3', type=str, help='GPU number')
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

############## Evaluate Classifier (LR, DT, RF, MLP, SVM) ##############
def evaluate_baseline_models(train_data, train_label, test_data, test_label, seed=25):
    baseline_results = []
    models = {
        "LogisticRegression": LogisticRegression(random_state=seed, max_iter=500),
        "DecisionTree": tree.DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(random_state=seed),
        "MLP": MLPClassifier(random_state=seed, max_iter=300),
        "SVM": svm.SVC(probability=True, random_state=seed)
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
    # gnn_model.plot_bars(instance_input, test_data_df.iloc[0], top_n)
    
    interpre_plot = f"/data/home/stateun/plot/Interpretability/{sampling}/"
    
    if not os.path.exists(interpre_plot):
        os.makedirs(interpre_plot)
    
    # interpre_data = f"{data_name}_interpretability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    # plot_filename = os.path.join(interpre_plot, interpre_data)
    # plt.savefig(plot_filename)
    # plt.close()
    
    ## Save the data
    # pred = gnn_model.predict(instance_input)
    # pred_filename = f"{data_name}_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
    
    # interpre_file = f"./data/interpretability/"
    # interpre_npz = f"{data_name}_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
    
    # if not os.path.exists(interpre_file):
    #     os.makedirs(interpre_file)
    
    # pred_filename = os.path.join(interpre_file, interpre_npz)
    
    # np.save(pred_filename, pred.cpu().detach().numpy())
    
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
    shap_plot = f"/data/home/stateun/plot/Interpretability/{sampling}/"
    
    if not os.path.exists(shap_plot):
        os.makedirs(shap_plot)
    
    shap_name = f"{data_name}_shap_gradient_top10_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    shap_plot_filename = os.path.join(shap_plot, shap_name)
    
    plt.savefig(shap_plot_filename, bbox_inches='tight')
    plt.close()

    print(f"[XAI] SHAP GradientExplainer top10 plot saved: {shap_plot_filename}")


############## Runner ##############
def run_experiment(args, now):
    """
    1. Load data from "./data"
    2. Feature Selection & Oversampling
    3. Train IGNnet (using validation to select best model)
    4. Evaluate on test set for IGNnet and for baseline models
    5. Run interpretability evaluation and return a list of result dictionaries (one for IGNnet and one for each baseline)
    """
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

    # Load Data
    full_path = os.path.join(args.file_path, args.data_name) + '.csv'
    load_data = pd.read_csv(full_path).astype(int)
    encoded_data = load_data.drop(['FraudFound_P'], axis=1)
    label_data = load_data['FraudFound_P']

    # Train / Validation / Test split
    train_data, left_out, train_label, y_left_out = train_test_split(
        encoded_data, label_data, test_size=0.2, random_state=args.seed
    )
    test_data, dev_data, test_label, dev_label = train_test_split(
        left_out, y_left_out, test_size=0.5, random_state=args.seed
    )

    train_set = pd.concat([train_data, train_label], axis=1)
    # Oversampling via CustomDataset
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

    # Compute adjacency matrix via data_preprocess
    graph_default = '/data/home/stateun/plot/graph'
    if not os.path.exists(graph_default):
        os.makedirs(graph_default)
    graph_name = f"{args.sampling}_{now}.png"
    graph_path = os.path.join(graph_default, graph_name)
    
    adj_matrix, index_to_name, name_to_index = dp.compute_adjacency_matrix(
        data=train_data, self_loop_weight=10, threshold=0.2
    )

    def categorize_feature(name: str) -> str:

        # 1) 
        if name == "AccidentArea":
            return "AccidentArea"
        elif name == "Sex":
            return "Sex"
        elif name == "Age":
            return "Age"
        elif name == "Fault":
            return "Fault"
        elif name == "VehiclePrice":
            return "VehiclePrice"
        elif name == "DriverRating":
            return "DriverRating"
        elif name == "AgeOfVehicle":
            return "AgeOfVehicle"
        elif name == "PoliceReportFiled":
            return "PoliceReportFiled"
        elif name == "AgentType":
            return "AgentType"
        elif name == "BasePolicy":
            return "BasePolicy"

        # 2) 
        elif name.startswith("Make_"):
            return "Make"
        elif name.startswith("MonthClaimed_"):
            return "MonthClaimed"
        elif name.startswith("MaritalStatus_"):
            return "MaritalStatus"
        elif name.startswith("PolicyType_"):
            return "PolicyType"
        elif name.startswith("VehicleCategory_"):
            return "VehicleCategory"
        elif name.startswith("RepNumber_"):
            return "RepNumber"
        elif name.startswith("Deductible_"):
            return "Deductible"
        elif name.startswith("PastNumberOfClaims_"):
            return "PastNumberOfClaims"
        elif name.startswith("AgeOfPolicyHolder_"):
            return "AgeOfPolicyHolder"
        elif name.startswith("NumberOfSuppliments_"):
            return "NumberOfSuppliments"
        elif name.startswith("AddressChange_Claim_"):
            return "AddressChange_Claim"
        elif name.startswith("NumberOfCars_"):
            return "NumberOfCars"
        elif name.startswith("Year_"):
            return "Year"
        elif "Days_Policy" in name or "Accident" in name:
            return "Accident_Policy"

        else:
            raise ValueError(f"Not Defined Feature Name : {name}")

    cat_to_nodes = defaultdict(list)
    for idx, feat_name in index_to_name.items():
        cat = categorize_feature(feat_name)
        cat_to_nodes[cat].append(idx)

    cat_list = list(cat_to_nodes.keys())
    cat_index = {cat: i for i, cat in enumerate(cat_list)}

    adj_cat = np.zeros((len(cat_list), len(cat_list)))
    for cat1, nodes1 in cat_to_nodes.items():
        for cat2, nodes2 in cat_to_nodes.items():
            total_weight = 0
            for n1 in nodes1:
                for n2 in nodes2:
                    total_weight += adj_matrix[n1, n2]
            i, j = cat_index[cat1], cat_index[cat2]
            adj_cat[i, j] = total_weight
            
    G_cat = nx.from_numpy_array(adj_cat)

    mapping_cat = {i: cat_list[i] for i in range(len(cat_list))}
    G_cat = nx.relabel_nodes(G_cat, mapping_cat)

    color_map = {
        "AccidentArea": "tab:blue",
        "Sex": "tab:orange",
        "Age": "tab:green",
        "Fault": "tab:red",
        "VehiclePrice": "tab:purple",
        "DriverRating": "tab:brown",
        "AgeOfVehicle": "tab:pink",
        "PoliceReportFiled": "tab:olive",
        "AgentType": "tab:gray",
        "BasePolicy": "tab:cyan",

        "Make": "tab:blue",
        "MonthClaimed": "tab:orange",
        "MaritalStatus": "tab:green",
        "PolicyType": "tab:red",
        "VehicleCategory": "tab:purple",
        "RepNumber": "tab:brown",
        "Deductible": "tab:pink",
        "PastNumberOfClaims": "tab:olive",
        "AgeOfPolicyHolder": "tab:gray",
        "NumberOfSuppliments": "tab:cyan",
        "AddressChange_Claim": "tab:olive",
        "NumberOfCars": "tab:green",
        "Year": "tab:purple",
        "Accident_Policy": "tab:red"
    }


    node_colors_cat = []
    for node in G_cat.nodes():
        node_colors_cat.append(color_map[node] if node in color_map else "tab:gray")

    plt.figure(figsize=(10, 10))
    pos_cat = nx.spring_layout(G_cat, seed=args.seed)
    nx.draw_networkx_nodes(G_cat, pos_cat, node_color=node_colors_cat, node_size=2000, alpha=0.7)
    nx.draw_networkx_edges(G_cat, pos_cat, alpha=0.5, width=1.2)
    nx.draw_networkx_labels(G_cat, pos_cat, font_size=10)
    plt.title("Graph Representation of Tabular Data (Merged by Category)")
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

    # Train IGNnet model – using validation to select best model
    gnn_model = train_model(input_dim, num_features, adj_matrix, index_to_name,
                              train_dataloader, val_dataloader,
                              data_name=args.data_name,
                              now = now,
                              loss = args.loss,
                              num_classes=2,
                              num_epochs=args.epochs,
                              learning_rate=args.lr,
                              normalize_adj=False)
    
    # Evaluate IGNnet on the test set
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
    
    # normalized_train_data = dp.min_max_normalize(train_data, train_data)
    run_interpretability_evaluation(gnn_model, test_tensor, adj_matrix, index_to_name, device, args.data_name, 
                                    sampling = args.sampling)
    
    all_results = [deep_results] + baseline_results
    return all_results


def main():
    args, _ = _parse_args()
    all_exp_results = []
    for dataset in ['fraud_data']:
        for batch_size in [64, 128, 256]:
            for epoch in [50, 100, 150, 200]:
                for sampling in ['smote', 'borderline-smote', 'adasyn', 'over-random']:
                    for alpha in [0.01, 0.03, 0.05, 0.1]:
                        for lr in [1e-2, 1e-4, 2e-2, 2e-4]:
                            for loss in ['Original']:
                                now = datetime.now().strftime("%H-%M-%S")
                                args.alpha = alpha
                                args.sampling = sampling
                                args.data_name = dataset
                                args.batch_size = batch_size
                                args.epochs = epoch
                                args.lr = lr
                                args.loss = loss
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
                                all_exp_results.extend(exp_results)
    results_df = pd.DataFrame(all_exp_results, columns=["dataset", "model", "AUC", "ACC", "Precision", "Recall", "F1_score", "best_params"])
    results_csv_path = os.path.join(args.output, "test1_results.csv")
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    if os.path.exists(results_csv_path):
        results_df.to_csv(results_csv_path, mode='a', header=False, index=False)
        print(f"Results appended to {results_csv_path}")
    else:
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to {results_csv_path}")

if __name__ == '__main__':
    main()