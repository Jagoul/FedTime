#!/bin/bash

# FedTime Ablation Study Scripts
# This script runs ablation studies to analyze the impact of different components

echo "Starting FedTime Ablation Study..."

# Common parameters
seq_len=96
pred_len=720
dataset="ETTh1"
data_path="ETTh1.csv"
root_path="./dataset/ETT-small/"
num_clients=10
num_rounds=50
local_epochs=5

# 1. Baseline: FedTime with all components
echo "=== 1. Full FedTime (Baseline) ==="
python -u run_federated.py \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id FedTime_Full_${dataset}_${seq_len}_${pred_len} \
    --model FedTime \
    --data $dataset \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --num_clients $num_clients \
    --num_rounds $num_rounds \
    --local_epochs $local_epochs \
    --use_clustering 1 \
    --num_clusters 3 \
    --use_peft 1 \
    --peft_method qlora \
    --use_dpo 1 \
    --des 'Full'

# 2. Without Clustering
echo "=== 2. FedTime without Clustering ==="
python -u run_federated.py \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id FedTime_NoClustering_${dataset}_${seq_len}_${pred_len} \
    --model FedTime \
    --data $dataset \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --num_clients $num_clients \
    --num_rounds $num_rounds \
    --local_epochs $local_epochs \
    --use_clustering 0 \
    --use_peft 1 \
    --peft_method qlora \
    --use_dpo 1 \
    --des 'NoClustering'

# 3. Without PEFT
echo "=== 3. FedTime without PEFT ==="
python -u run_federated.py \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id FedTime_NoPEFT_${dataset}_${seq_len}_${pred_len} \
    --model FedTime \
    --data $dataset \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --num_clients $num_clients \
    --num_rounds $num_rounds \
    --local_epochs $local_epochs \
    --use_clustering 1 \
    --num_clusters 3 \
    --use_peft 0 \
    --use_dpo 1 \
    --des 'NoPEFT'

# 4. Without DPO
echo "=== 4. FedTime without DPO ==="
python -u run_federated.py \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id FedTime_NoDPO_${dataset}_${seq_len}_${pred_len} \
    --model FedTime \
    --data $dataset \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --num_clients $num_clients \
    --num_rounds $num_rounds \
    --local_epochs $local_epochs \
    --use_clustering 1 \
    --num_clusters 3 \
    --use_peft 1 \
    --peft_method qlora \
    --use_dpo 0 \
    --des 'NoDPO'

# 5. Different number of clusters
echo "=== 5. FedTime with 2 clusters ==="
python -u run_federated.py \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id FedTime_2Clusters_${dataset}_${seq_len}_${pred_len} \
    --model FedTime \
    --data $dataset \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --num_clients $num_clients \
    --num_rounds $num_rounds \
    --local_epochs $local_epochs \
    --use_clustering 1 \
    --num_clusters 2 \
    --use_peft 1 \
    --peft_method qlora \
    --use_dpo 1 \
    --des '2Clusters'

echo "=== 6. FedTime with 5 clusters ==="
python -u run_federated.py \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id FedTime_5Clusters_${dataset}_${seq_len}_${pred_len} \
    --model FedTime \
    --data $dataset \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --num_clients $num_clients \
    --num_rounds $num_rounds \
    --local_epochs $local_epochs \
    --use_clustering 1 \
    --num_clusters 5 \
    --use_peft 1 \
    --peft_method qlora \
    --use_dpo 1 \
    --des '5Clusters'

# 6. Different PEFT methods
echo "=== 7. FedTime with LoRA (no quantization) ==="
python -u run_federated.py \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id FedTime_LoRA_${dataset}_${seq_len}_${pred_len} \
    --model FedTime \
    --data $dataset \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --num_clients $num_clients \
    --num_rounds $num_rounds \
    --local_epochs $local_epochs \
    --use_clustering 1 \
    --num_clusters 3 \
    --use_peft 1 \
    --peft_method lora \
    --use_dpo 1 \
    --des 'LoRA'

# 7. Different aggregation methods
echo "=== 8. FedTime with FedAdam ==="
python -u run_federated.py \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id FedTime_FedAdam_${dataset}_${seq_len}_${pred_len} \
    --model FedTime \
    --data $dataset \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --num_clients $num_clients \
    --num_rounds $num_rounds \
    --local_epochs $local_epochs \
    --aggregation_method fedadam \
    --server_lr 0.01 \
    --use_clustering 1 \
    --num_clusters 3 \
    --use_peft 1 \
    --peft_method qlora \
    --use_dpo 1 \
    --des 'FedAdam'

# 8. Different number of clients
echo "=== 9. FedTime with 5 clients ==="
python -u run_federated.py \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id FedTime_5Clients_${dataset}_${seq_len}_${pred_len} \
    --model FedTime \
    --data $dataset \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --num_clients 5 \
    --num_rounds $num_rounds \
    --local_epochs $local_epochs \
    --use_clustering 1 \
    --num_clusters 2 \
    --use_peft 1 \
    --peft_method qlora \
    --use_dpo 1 \
    --des '5Clients'

echo "=== 10. FedTime with 20 clients ==="
python -u run_federated.py \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id FedTime_20Clients_${dataset}_${seq_len}_${pred_len} \
    --model FedTime \
    --data $dataset \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --num_clients 20 \
    --num_rounds $num_rounds \
    --local_epochs $local_epochs \
    --use_clustering 1 \
    --num_clusters 4 \
    --use_peft 1 \
    --peft_method qlora \
    --use_dpo 1 \
    --des '20Clients'

echo "=== Ablation Study Completed ==="
