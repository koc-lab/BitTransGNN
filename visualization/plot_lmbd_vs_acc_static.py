import pandas as pd
import matplotlib.pyplot as plt

base_dir = '/Users/eralpkumbasar/Documents/Master Research/1-bit LLM/Kod/all_codes/BertGCN/'
dataset_name_list = ["mr", "ohsumed", "cola", "stsb", "rte", "mrpc"]

def read_avg_results(base_dir, dataset):
    if dataset in ["mr", "ohsumed", "rte", "mrpc"]:
        desired_metrics = ["best_test_accuracy", "best_test_accuracy.1"]
    elif dataset == "stsb":
        desired_metrics = ["best_test_pearson_corr", "best_test_pearson_corr.1"]
    elif dataset == "cola":
        desired_metrics = ["best_test_matthews_corr", "best_test_matthews_corr.1"]
    # Load the CSV file
    bitbertgcn_file_path = base_dir + 'bitbertgcn_stats_seed.csv'
    df_bitbertgcn = pd.read_csv(bitbertgcn_file_path)

    bitbert_file_path = base_dir + 'bitbert_stats_seed.csv'
    df_bitbert = pd.read_csv(bitbert_file_path)

    bitbertgcn_config_keys = ["dataset_name", "num_states", "lmbd"]
    filtered_df_bitbertgcn = df_bitbertgcn[(df_bitbertgcn['dataset_name'] == dataset) & 
                                            (df_bitbertgcn['bert_quant_type'] == 'QAT') & 
                                            (df_bitbertgcn['bert_pre_model'] == 'bert-base-uncased') &
                                            (df_bitbertgcn["lmbd"].isin([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])) &
                                            (df_bitbertgcn['train_type'] == 'static')]

    filtered_df_bitbertgcn = filtered_df_bitbertgcn[bitbertgcn_config_keys + desired_metrics]

    bitbert_config_keys = ["dataset_name", "model_configs|num_states"]
    # Filter rows where dataset_name is "mr" or "ohsumed", bert_quant_type is "QAT", and train_type is "static"
    filtered_df_bitbert = df_bitbert[(df_bitbert['dataset_name'] == dataset) & 
                                        (df_bitbert['model_configs|quantize_embeddings'] == False) & 
                                        (df_bitbert['model_configs|num_bits_act'] == 8.0) &
                                        (df_bitbert['model_configs|num_states'].isin([2.0, 3.0, 5.0])) &
                                        (df_bitbert["bert_pre_model"] == "bert-base-uncased")]

    filtered_df_bitbert = filtered_df_bitbert[bitbert_config_keys + desired_metrics]

    filtered_df_bitbert.rename(columns={"model_configs|num_states": "num_states"}, inplace=True)

    bitbert_lmbds = []
    for _ in range(len(filtered_df_bitbert)):
        bitbert_lmbds += [0.0]

    filtered_df_bitbert["lmbd"] = bitbert_lmbds
    filtered_df_bitbert = filtered_df_bitbert[bitbertgcn_config_keys + desired_metrics]
    if dataset in ["mr", "ohsumed"]:
        filtered_df_bitbertgcn = pd.concat([filtered_df_bitbertgcn, filtered_df_bitbert])
    desired_metrics = ["test_avg", "test_std"]
    if dataset in ["mr", "ohsumed", "rte", "mrpc"]:
        filtered_df_bitbertgcn.rename(columns={"best_test_accuracy": "test_avg", 
                                            "best_test_accuracy.1": "test_std"}, inplace=True)
    elif dataset == "stsb":
        filtered_df_bitbertgcn.rename(columns={"best_test_pearson_corr": "test_avg", 
                                            "best_test_pearson_corr.1": "test_std"}, inplace=True)
    elif dataset == "cola":
        filtered_df_bitbertgcn.rename(columns={"best_test_matthews_corr": "test_avg", 
                                            "best_test_matthews_corr.1": "test_std"}, inplace=True)
    filtered_df_bitbertgcn = filtered_df_bitbertgcn.sort_values(by=["dataset_name", "num_states", "lmbd"]).reset_index(drop=True)[bitbertgcn_config_keys + desired_metrics]
    for metric in ["num_states", "lmbd", "test_avg", "test_std"]:
        filtered_df_bitbertgcn[metric] = pd.to_numeric(filtered_df_bitbertgcn[metric], errors='coerce')
    return(filtered_df_bitbertgcn)

def plot_results(base_dir, dataset):
    print(f"dataset: {dataset}")
    filtered_df = read_avg_results(base_dir, dataset)
    num_states_list = [2.0, 3.0, 5.0]
    label_list = ["1-bit", "1.58-bit", "2.32-bit"]
    lmbd_list = filtered_df["lmbd"].unique()
    #acc_list = np.zeros((len(lmbd_list), len(num_states_list)))
    #std_list = np.zeros((len(lmbd_list), len(num_states_list)))
    acc_list, std_list = [], []
    for num_states in num_states_list:
        acc_list.append(filtered_df[filtered_df["num_states"] == num_states]["test_avg"].values)
        std_list.append(filtered_df[filtered_df["num_states"] == num_states]["test_std"].values)

    formats = ['-o', '--s', '-.^']

    plt.figure(figsize=(8,6))

    # Plot each column
    for i in range(len(acc_list)):
        #plt.errorbar(lmbd_list, acc_list[i], yerr=std_list[i], fmt=formats[i], capsize=5, label=num_states_list[i])
        line, _, _ = plt.errorbar(lmbd_list, acc_list[i], fmt=formats[i], capsize=5, label=label_list[i])
        max_acc = max(acc_list[i])
        max_index = list(acc_list[i]).index(max_acc)
        plt.plot(lmbd_list[max_index], max_acc, '*', color=line.get_color(), markersize=12)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.grid(True)

    plt.legend(label_list)
    plt.xticks(ticks=lmbd_list, labels=lmbd_list, fontweight="bold")
    plt.yticks(fontweight="bold")
    plt.xlabel(r'$\mathbf{\lambda}$',fontsize=15,labelpad=5,fontweight="bold")
    #plt.title(dataset.upper(),fontweight="bold")
    plt.legend(prop={'weight': 'bold'})
    plt.ylabel(r'Test Accuracy ($\%$)',fontsize=15,labelpad=5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"./lmbd_plot/{dataset}_lmbd_vs_acc.png")

for dataset in dataset_name_list:
#for dataset in ["mr"]:
    plot_results(base_dir, dataset)