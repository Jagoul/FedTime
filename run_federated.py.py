import argparse
import os
import torch
import random
import numpy as np
from exp.exp_federated import Exp_Federated

def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='FedTime: Federated Training')

    # Basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='FedTime',
                        help='model name, options: [FedTime]')

    # Data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # Federated Learning Parameters
    parser.add_argument('--num_clients', type=int, default=10, help='number of federated clients')
    parser.add_argument('--num_rounds', type=int, default=100, help='number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=5, help='local training epochs per round')
    parser.add_argument('--client_fraction', type=float, default=1.0, help='fraction of clients participating per round')
    parser.add_argument('--aggregation_method', type=str, default='fedavg', 
                        help='aggregation method: fedavg, fedadam, fedopt')
    
    # Clustering parameters
    parser.add_argument('--use_clustering', type=int, default=1, help='use K-means clustering; True 1 False 0')
    parser.add_argument('--num_clusters', type=int, default=3, help='number of clusters for K-means')
    parser.add_argument('--clustering_features', type=str, default='data_stats', 
                        help='features for clustering: data_stats, model_params, mixed')
    
    # Model define (same as centralized)
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

    # FedTime specific parameters
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    
    # LLM parameters
    parser.add_argument('--llm_model', type=str, default='LLaMA', help='LLM model')
    parser.add_argument('--llm_dim', type=int, default=4096, help='LLM model dimension')
    parser.add_argument('--llm_layers', type=int, default=32, help='LLM model layers')
    
    # PEFT parameters
    parser.add_argument('--use_peft', type=int, default=1, help='use PEFT; True 1 False 0')
    parser.add_argument('--peft_method', type=str, default='qlora', help='PEFT method: lora, qlora')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    
    # DPO parameters
    parser.add_argument('--use_dpo', type=int, default=1, help='use DPO; True 1 False 0')
    parser.add_argument('--dpo_beta', type=float, default=0.1, help='DPO beta parameter')

    # Optimization
    parser.add_argument('--batch_size', type=int, default=128, help='