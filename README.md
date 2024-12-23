# Quantized Transformers with Graph Neural Networks

This repository contains the implementation of "Quantized Transformers with Graph Neural Networks" about to be submitted to IEEE Transactions on Signal Processing (TSP). 

## Introduction

In this work, we propose a method to improve the performance of transformers quantized to low-bit precision. Our main method, `BitBERTGCN`, trains a quantized transformer jointly with a Graph Neural Network (GNN) to improve quantized model performance. We additionally propose two extensions that eliminate the shortcoming of `BitBERTGCN` by seperating the quantized transformer from GNNs after training. These extensions allow to use the quantized transformer solitarily. `BitBERTGCN` methods significantly improve quantized transformer performance while maintaining the efficiency advantages in inference time.

## Installation

1. Cloning the repository
   ```bash
   git clone https://github.com/EralpK/BitBERTGCN.git

2. Installing the required dependencies

   We recommend to create a Conda environment or a virtual environment and install the dependencies to the environment. 
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, the dependencies can also be installed from the .yml file:
   ```bash
   conda env create -f environment.yml
   ```

## Running the Code
   The main folder contains `bitbertgcn` and `bitbertgcn_calcs` folders. `bitbertgcn` folder contains the scripts written to implement the methods introduced in the manuscript. `bitbertgcn_calcs` contains sample notebooks that calculate the theoretical memory and energy consumption of different quantized methods.
   To run the method, first navigate to the `bitbertgcn` directory:
   ```bash
   cd bitbertgcn
   ```
   Notebook files that investigate the memory and energy efficiency of different transformer models can be inspected within the `bitbertgcn_calcs` folder.

## Dataset Preprocessing
   For 20NG, MR, R8, R52 and Ohsumed datasets, we first clean and tokenize the raw text data. 
   ```bash
   python remove_words.py
   ```
   To construct the raw text files for the GLUE datasets, prepare_data_glue.py script can be used.
   ```bash
   python prepare_data_glue.py
   ```
   After the preprocessing, the text graphs for each dataset are constructed:
   ```bash
   python build_graph.py
   ```
   The text graphs constructed through `build_graph.py` contain all train, test, and validation nodes. In case one wants to operate methods only over the training nodes, the following command should be executed:
   ```bash
   python build_graph_inductive.py
   ```

## Methods
   Our work is a composition of four methods. The four methods are contained within the `/methods` directory. We list the methods in the table below:
   | Method    | Description |
   | -------- | ------- |
   | bitbert  | Fine-tunes full-precision and quantized transformer models before integrating them into the main architecture.|
   | bitbertgcn | Trains quantized transformers jointly with GNNs in different configurations.|
   | direct-seperation    | Seperates quantized transformer model from the joint architecture after training for inference.|
   | knowledge-distillation    | Compresses the knowledge of the joint architecture into a quantized transformer model through knowledge distillation.|

   Each of the `bitbert`, `bitbertgcn` and `knowledge-distillation` methods contain `train.py`, `run_sweep.py` and `inference.py` scripts. direct-seperation involves no additional training and only applies inference. For that reason, it only contains the `inference.py` script.

   `train.py` script is used to train the method based on the configurations defined in `configs/train_config.py`. `sweep.py` conducts training over the ranges of hyperparameters defined in `configs/sweep_config.yaml`. The trained models can be used for inference using the script in `inference.py` based on the configuration in `configs/sweep_config.yaml`. 

   To use each script, 
   ```bash
   python -m methods.{framework}.{mode}
   ```
   where `{framework}` refers to one of the four methods being used and `{mode}` defines whether the script is used for training, testing, or sweeping through hyperparameters.
#
### Fine-Tuning Transformer Models with Quantization
   We fine-tune quantized transformer models before integrating them into our architecture. The fine-tuning is conducted through the following code:
   ```bash
   python -m methods.bitbert.train
   ```

   The `save_ckpt` variable should be set to `True` in the `configs/train_config.yaml` file if one wants to save model checkpoint for later use in the joint architecture.
#
### BitBERTGCN: Training Quantized Transformers with GNNs
   We combine quantized transformers with GNNs during training to improve their performance and enhance their predictions. We use a Graph Convolutional Network (GCN) as the GNN companion. We interpolate the outputs of the quantized transformer and the GNN during training. 
   <!-- Uncomment this part after reviewing stage is completed -->
   <!-- The outputs of the quantized transformer and the GNN are interpolated through the following equation:
   $$\mathbf{Z} = \lambda \mathbf{Z}_{\text{GCN}} + (1-\lambda) \mathbf{Z}_{\text{BERT}}$$  -->
   We propose `Static` and `Dynamic` variants of BitBERTGCN methods. `Static` methods use quantized transformer models only for inference, while `Dynamic` methods jointly train the two models. The `joint_training` variable should be set to `True` to train the `Dynamic` variant, and it should be set to `False` to train a `Static` `BitBERTGCN` method. The following line of code should be run to train BitBERTGCN methods:
   ```bash
   python -m methods.bitbertgcn.train
   ```

   Note that the parameters of a fine-tuned quantized transformer should be saved either within the local machine or an external source to be able to operate the joint architecture.

   We combine quantization and fine-tuning in two different stages. Quantization-Aware Fine-Tuning (QAFT) first quantizes the transformer parameters and then fine-tunes the model over the task. Post Fine-Tuning Quantization (PFTQ) conducts quantization after fine-tuning is completed. 

   We use both QAFT and PFTQ while integrating quantized transformer models into `Dynamic` training. We only integrate QAFT methods into `Static` `BitBERTGCN` training. The fine-tuning method can be chosen by changing the `bert_quant_type` file in the configuration file.

   Different $\lambda$ values can be searched using the `run_sweep.py` script. 

   For use in `direct-seperation` and `knowledge-disillation` methods, the `save_ckpt` variable should be set to `True`.
#
### Direct Seperation (DS)
   After `BitBERTGCN` training, `DS` seperates the quantized transformer and its classifier from the GCN after training and uses them for inference, without GCN's assistance during inference time. 
   <!-- Uncomment this part after reviewing stage is completed -->
   <!-- The output is as following:
   $$\mathbf{Z} = \mathbf{Z}_{\text{BERT}}$$  -->
   To use the quantized transformer that is jointly trained with the GNN solitarily in inference time:
   ```bash
   python -m methods.direct-seperation.inference
   ```
#
### Seperation Through Knowledge Distillation (KD)
   `KD` compresses the knowledge of the `BitBERTGCN` model into a solitary quantized transformer through distillation. A `BitBERTGCN` model is used as the teacher and a `BitBERT` model is used as the student model.

   To train the student `BitBERT`:
   ```bash
   python -m methods.knowledge_distillation.train
   ```

   We conduct offline distillation. The distillation type can be modified from the configuration file. The sweeping script can be used to search over the distillation parameters $\alpha_d$ and $T$:
   ```bash
   python -m methods.knowledge_distillation.run_sweep
   ```

## Results
   Results will be made public upon the completion of the reviewing stage by IEEE TSP. 

## Acknowledgements 
   - The data preprocessing and graph construction scripts are adapted from [TextGCN](https://github.com/yao8839836/text_gcn) and [BertGCN](https://github.com/ZeroRin/BertGCN) repositories.
   - We acknowledge the guidance we received from the implementation of [BitNet](https://github.com/microsoft/BitNet) during the construction of our quantization module.

