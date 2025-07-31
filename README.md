# FedTime: A Federated Large Language Model for Long-Term Time Series Forecasting

[![arXiv](https://img.shields.io/badge/arXiv-2407.20503-b31b1b.svg)](https://arxiv.org/abs/2407.20503)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official PyTorch implementation of **FedTime**, a federated large language model framework for long-term time series forecasting that preserves data privacy while achieving superior performance.

## Paper
- **[(https://arxiv.org/abs/2407.20503)]** Paper accepted and available on arXiv

## 📖 Abstract

Long-term time series forecasting in centralized environments poses unique challenges regarding data privacy, communication overhead, and scalability. To address these challenges, we propose **FedTime**, a federated large language model (LLM) tailored for long-range time series prediction. 

Our key contributions include:
- 🏗️ **Federated LLM Framework**: First federated learning approach using LLMs for time series forecasting
- 🎯 **K-means Clustering**: Pre-processing step to partition edge devices into clusters for focused training
- 🔧 **Parameter-Efficient Fine-tuning**: QLoRA-based approach reducing communication overhead
- 📊 **Superior Performance**: Outperforms state-of-the-art methods on multiple benchmark datasets
- 🔒 **Privacy Preservation**: Maintains data privacy while enabling collaborative learning

## 🏆 Key Results

| Dataset | Method | MSE (T=720) | MAE (T=720) | Improvement |
|---------|---------|-------------|-------------|-------------|
| Traffic | LLM4TS | 0.437 | 0.292 | - |
| Traffic | **FedTime** | **0.369** | **0.239** | **15.56%** ↓ |
| Electricity | LLM4TS | 0.220 | 0.292 | - |
| Electricity | **FedTime** | **0.176** | **0.288** | **20.0%** ↓ |
| ETTm1 | LLM4TS | 0.408 | 0.419 | - |
| ETTm1 | **FedTime** | **0.328** | **0.373** | **10.98%** ↓ |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FedTime.git
cd FedTime

# Create conda environment
conda create -n fedtime python=3.8
conda activate fedtime

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

Download and prepare the datasets:

```bash
# Download datasets
bash scripts/download_datasets.sh

# Prepare federated data splits
python data_provider/data_factory.py --data ETTh1 --federated --num_clients 10
```

### Training

#### Centralized Training (Baseline)
```bash
python run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model FedTime \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1
```

#### Federated Training
```bash
python run_federated.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id FedETTh1_96_192 \
  --model FedTime \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --num_clients 10 \
  --num_rounds 100 \
  --local_epochs 5 \
  --use_clustering 1 \
  --num_clusters 3 \
  --use_peft 1 \
  --peft_method qlora \
  --des 'FedExp' \
  --itr 1
```

### Evaluation

```bash
# Evaluate trained model
python run_longExp.py \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model FedTime \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --des 'Exp' \
  --itr 1
```

## 📁 Repository Structure

```
FedTime/
├── 📁 data_provider/           # Data loading and preprocessing
│   ├── data_factory.py
│   ├── data_loader.py
│   └── federated_data.py
├── 📁 dataset/                 # Dataset storage
├── 📁 exp/                     # Experiment runners
│   ├── exp_basic.py
│   └── exp_federated.py
├── 📁 layers/                  # Model layers and components
│   ├── Transformer_EncDec.py
│   ├── SelfAttention_Family.py
│   └── Embed.py
├── 📁 models/                  # Model implementations
│   ├── FedTime.py
│   └── __init__.py
├── 📁 scripts/                 # Training and evaluation scripts
│   ├── FedTime/
│   │   ├── federated_training.sh
│   │   └── centralized_baseline.sh
│   └── download_datasets.sh
├── 📁 federated/               # Federated learning components
│   ├── client.py
│   ├── server.py
│   ├── aggregation.py
│   └── clustering.py
├── 📁 utils/                   # Utility functions
│   ├── tools.py
│   ├── metrics.py
│   └── timefeatures.py
├── run_longExp.py             # Main training script
├── run_federated.py           # Federated training script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🔧 Model Architecture

FedTime consists of several key components:

1. **Channel Independence**: Treats each time series variable separately
2. **Patching**: Divides time series into smaller patches for efficient processing
3. **LLaMA-2 Encoder**: Uses pre-trained LLaMA-2-7B as the backbone
4. **QLoRA Fine-tuning**: Parameter-efficient fine-tuning with quantization
5. **Direct Preference Optimization (DPO)**: Aligns model with time series data
6. **Federated Aggregation**: Combines local model updates while preserving privacy

## 📊 Experiments

### Reproducing Paper Results

We provide scripts to reproduce all experimental results from the paper:

```bash
# Run all experiments
bash scripts/run_all_experiments.sh

# Individual dataset experiments
bash scripts/FedTime/ETTh1.sh
bash scripts/FedTime/ETTh2.sh
bash scripts/FedTime/ETTm1.sh
bash scripts/FedTime/ETTm2.sh
bash scripts/FedTime/Weather.sh
bash scripts/FedTime/Traffic.sh
bash scripts/FedTime/Electricity.sh
```

### Ablation Studies

```bash
# Without clustering
python run_federated.py --use_clustering 0 --des 'NoClustering'

# Without PEFT
python run_federated.py --use_peft 0 --des 'NoPEFT'

# Different number of clusters
python run_federated.py --num_clusters 2 --des 'Clusters2'
python run_federated.py --num_clusters 5 --des 'Clusters5'
```

## 📈 Communication Overhead Analysis

Analyze communication costs:

```bash
python analyze_communication.py \
  --model FedTime \
  --num_clients 10 \
  --num_rounds 100 \
  --save_results ./results/communication_analysis.json
```

## 🔍 Supported Datasets

- **ETT (Electricity Transformer Temperature)**: ETTh1, ETTh2, ETTm1, ETTm2
- **Weather**: Meteorological data with 21 indicators
- **Traffic**: Road occupancy rates from California
- **Electricity**: Electricity consumption data
- **ACN**: Adaptive charging network dataset for EV charging

## 🛠️ Customization

### Adding New Datasets

1. Implement data loader in `data_provider/data_loader.py`
2. Add dataset configuration in `data_provider/data_factory.py`
3. Create federated splits using `data_provider/federated_data.py`

### Implementing New Aggregation Methods

1. Add aggregation function in `federated/aggregation.py`
2. Update server logic in `federated/server.py`
3. Test with `run_federated.py --aggregation_method your_method`

## Citation

If you find our work useful, please cite:

```bibtex
@incollection{abdel2024federated,
  title={A federated large language model for long-term time series forecasting},
  author={Abdel-Sater, Raed and Ben Hamza, A},
  booktitle={ECAI 2024},
  pages={2452--2459},
  year={2024},
  publisher={IOS Press}
}
```

## 📞 Contact

- **Raed Abdel-Sater**: raed.abdelsater@mail.concordia.ca
- **A. Ben Hamza**: hamza@ciise.concordia.ca

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- This work was supported by the Discovery Grants Program of NSERC Canada under grant RGPIN-2024-04291
- Thanks to the authors of PatchTST and LLaMA for their foundational work
- Dataset providers: ETT, Weather, Traffic, Electricity, and ACN datasets

## 🔗 Related Work

- [PatchTST](https://github.com/yuqinie98/PatchTST): A Time Series is Worth 64 Words
- [LLaMA](https://github.com/facebookresearch/llama): Large Language Model Meta AI
- [QLoRA](https://github.com/artidoro/qlora): Efficient Finetuning of Quantized LLMs

---

⭐ **Star this repository if you find it helpful!**
# FedTime : A-Federated-Large-Language-Model-for-Long-Term-Time-Series-Forecasting
The official PyTorch implementation of "A Federated Large Language Model for Long-Term Time Series Forecasting" by Raed Abdel-Sater and A. Ben Hamza from Concordia University. The paper introduces FedTime, a novel federated learning framework that leverages Large Language Models (LLMs) for privacy-preserving, collaborative time series forecasting
