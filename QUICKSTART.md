# FedTime Quick Start Guide

This guide will help you get started with FedTime quickly.

## 🚀 Quick Setup (5 minutes)

### Option 1: Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/FedTime.git
cd FedTime

# Build and run with Docker Compose
docker-compose up -d fedtime

# Enter the container
docker exec -it fedtime_container bash

# Download sample dataset
bash scripts/download_datasets.sh

# Run a quick test
python run_federated.py \
  --is_training 1 \
  --data ETTh1 \
  --model FedTime \
  --seq_len 96 \
  --pred_len 96 \
  --num_clients 5 \
  --num_rounds 10 \
  --des 'QuickTest'
```

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/FedTime.git
cd FedTime

# Create environment
conda create -n fedtime python=3.8
conda activate fedtime

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Download datasets
bash scripts/download_datasets.sh

# Quick test
python run_federated.py --is_training 1 --data ETTh1 --model FedTime --num_rounds 5 --des 'Test'
```

## 📊 Run Your First Experiment

### 1. Federated Training on ETTh1

```bash
python run_federated.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id MyFirstFedTime \
  --model FedTime \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --num_clients 10 \
  --num_rounds 50 \
  --local_epochs 5 \
  --use_clustering 1 \
  --num_clusters 3 \
  --use_peft 1 \
  --des 'MyFirstExp'
```

### 2. Compare with Centralized Training

```bash
python run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id CentralizedBaseline \
  --model FedTime \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --des 'Centralized'
```

### 3. Analyze Communication Overhead

```bash
python analyze_communication.py \
  --num_clients 10 \
  --num_rounds 50 \
  --output_dir ./my_analysis
```

## 📈 Understanding Results

After training, you'll find results in:

```
results/
├── Fed_MyFirstFedTime_FedTime_ETTh1_M_96_192_10_50_5_MyFirstExp_0/
│   ├── training_history.json          # Training metrics per round
│   ├── communication_overhead.json    # Communication statistics
│   └── metrics.npy                   # Final test metrics [MAE, MSE, RMSE, MAPE, MSPE]
└── ...
```

### Key Metrics to Check:

1. **Final Performance**: Check `metrics.npy` for MAE and MSE
2. **Communication Efficiency**: Look at `communication_overhead.json`
3. **Training Progress**: Plot data from `training_history.json`

## 🔧 Common Configurations

### Small Scale (Quick Testing)
```bash
--num_clients 5 --num_rounds 20 --local_epochs 3 --num_clusters 2
```

### Medium Scale (Paper Results)
```bash
--num_clients 10 --num_rounds 100 --local_epochs 5 --num_clusters 3
```

### Large Scale (Production)
```bash
--num_clients 50 --num_rounds 200 --local_epochs 5 --num_clusters 10
```

## 🛠️ Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size or use gradient checkpointing
```bash
--batch_size 64  # Default is 128
```

### Issue: Slow Training
**Solution**: Use fewer clients or rounds for testing
```bash
--num_clients 3 --num_rounds 10
```

### Issue: Poor Performance
**Solution**: Check these parameters:
- Increase `local_epochs` (default: 5)
- Adjust `learning_rate` (default: 0.0001)
- Try different `num_clusters`

### Issue: Dataset Not Found
**Solution**: Download datasets
```bash
bash scripts/download_datasets.sh
```

## 📚 Next Steps

1. **Run Full Experiments**: Use `scripts/FedTime/federated_training.sh`
2. **Ablation Studies**: Use `scripts/FedTime/ablation_study.sh`
3. **Custom Datasets**: Modify `data_provider/data_loader.py`
4. **Hyperparameter Tuning**: Adjust learning rates, batch sizes, PEFT parameters

## 🆘 Getting Help

- **Issues**: Check GitHub Issues
- **Documentation**: See README.md for detailed information
- **Paper**: Read the [arXiv paper](https://arxiv.org/abs/2407.20503)

## 🎯 Key Commands Summary

```bash
# Quick federated training
python run_federated.py --is_training 1 --data ETTh1 --model FedTime --num_rounds 10

# Quick centralized training  
python run_longExp.py --is_training 1 --data ETTh1 --model FedTime

# Communication analysis
python analyze_communication.py --output_dir ./analysis

# Full paper experiments
bash scripts/FedTime/federated_training.sh

# Ablation studies
bash scripts/FedTime/ablation_study.sh
```

Happy experimenting with FedTime! 🚀
