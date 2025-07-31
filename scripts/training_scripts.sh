#!/bin/bash

# FedTime Federated Training Scripts
# This script runs federated experiments for all datasets mentioned in the paper

# Set common parameters
seq_len=96
pred_lens=(96 192 336 720)
num_clients=10
num_rounds=100
local_epochs=5

echo "Starting FedTime Federated Training Experiments..."

# ETTh1 Dataset
echo "=== Training on ETTh1 Dataset ==="
for pred_len in "${pred_lens[@]}"; do
    echo "ETTh1 - pred_len: $pred_len"
    python -u run_federated.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id FedETTh1_${seq_len}_${pred_len} \
        --model FedTime \
        --data ETTh1 \
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
        --des 'FedExp'
done

# ETTh2 Dataset
echo "=== Training on ETTh2 Dataset ==="
for pred_len in "${pred_lens[@]}"; do
    echo "ETTh2 - pred_len: $pred_len"
    python -u run_federated.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id FedETTh2_${seq_len}_${pred_len} \
        --model FedTime \
        --data ETTh2 \
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
        --des 'FedExp'
done

# ETTm1 Dataset
echo "=== Training on ETTm1 Dataset ==="
for pred_len in "${pred_lens[@]}"; do
    echo "ETTm1 - pred_len: $pred_len"
    python -u run_federated.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id FedETTm1_${seq_len}_${pred_len} \
        --model FedTime \
        --data ETTm1 \
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
        --des 'FedExp'
done

# ETTm2 Dataset
echo "=== Training on ETTm2 Dataset ==="
for pred_len in "${pred_lens[@]}"; do
    echo "ETTm2 - pred_len: $pred_len"
    python -u run_federated.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id FedETTm2_${seq_len}_${pred_len} \
        --model FedTime \
        --data ETTm2 \
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
        --des 'FedExp'
done

# Weather Dataset
echo "=== Training on Weather Dataset ==="
for pred_len in "${pred_lens[@]}"; do
    echo "Weather - pred_len: $pred_len"
    python -u run_federated.py \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id FedWeather_${seq_len}_${pred_len} \
        --model FedTime \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --num_clients $num_clients \
        --num_rounds $num_rounds \
        --local_epochs $local_epochs \
        --use_clustering 1 \
        --num_clusters 3 \
        --use_peft 1 \
        --peft_method qlora \
        --des 'FedExp'
done

# Traffic Dataset
echo "=== Training on Traffic Dataset ==="
for pred_len in "${pred_lens[@]}"; do
    echo "Traffic - pred_len: $pred_len"
    python -u run_federated.py \
        --is_training 1 \
        --root_path ./dataset/traffic/ \
        --data_path traffic.csv \
        --model_id FedTraffic_${seq_len}_${pred_len} \
        --model FedTime \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 862 \
        --dec_in 862 \
        --c_out 862 \
        --num_clients $num_clients \
        --num_rounds $num_rounds \
        --local_epochs $local_epochs \
        --use_clustering 1 \
        --num_clusters 3 \
        --use_peft 1 \
        --peft_method qlora \
        --des 'FedExp'
done

# Electricity Dataset
echo "=== Training on Electricity Dataset ==="
for pred_len in "${pred_lens[@]}"; do
    echo "Electricity - pred_len: $pred_len"
    python -u run_federated.py \
        --is_training 1 \
        --root_path ./dataset/electricity/ \
        --data_path electricity.csv \
        --model_id FedElectricity_${seq_len}_${pred_len} \
        --model FedTime \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --num_clients $num_clients \
        --num_rounds $num_rounds \
        --local_epochs $local_epochs \
        --use_clustering 1 \
        --num_clusters 3 \
        --use_peft 1 \
        --peft_method qlora \
        --des 'FedExp'
done

echo "=== All Federated Training Experiments Completed ==="