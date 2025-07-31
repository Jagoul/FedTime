#!/bin/bash

echo "=============================================="
echo "    FedTime Dataset Download Script"
echo "=============================================="

# Create dataset directories
echo "Creating dataset directories..."
mkdir -p dataset/ETT-small
mkdir -p dataset/weather
mkdir -p dataset/traffic
mkdir -p dataset/electricity
mkdir -p dataset/ACN

# Function to download file with retry
download_with_retry() {
    local url=$1
    local output=$2
    local max_attempts=3
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt: Downloading $(basename $output)..."
        if wget -O "$output" "$url" 2>/dev/null; then
            echo "✓ Successfully downloaded $(basename $output)"
            return 0
        else
            echo "✗ Failed to download $(basename $output) (attempt $attempt)"
            attempt=$((attempt + 1))
            if [ $attempt -le $max_attempts ]; then
                echo "Retrying in 5 seconds..."
                sleep 5
            fi
        fi
    done
    
    echo "✗ Failed to download $(basename $output) after $max_attempts attempts"
    return 1
}

# Download ETT datasets
echo ""
echo "Downloading ETT (Electricity Transformer Temperature) datasets..."
echo "--------------------------------------------------------------"

ETT_BASE_URL="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"

download_with_retry "$ETT_BASE_URL/ETTh1.csv" "dataset/ETT-small/ETTh1.csv"
download_with_retry "$ETT_BASE_URL/ETTh2.csv" "dataset/ETT-small/ETTh2.csv"
download_with_retry "$ETT_BASE_URL/ETTm1.csv" "dataset/ETT-small/ETTm1.csv"
download_with_retry "$ETT_BASE_URL/ETTm2.csv" "dataset/ETT-small/ETTm2.csv"

# Create a simple weather dataset (synthetic for demo)
echo ""
echo "Creating sample Weather dataset..."
echo "--------------------------------"

cat > dataset/weather/weather.csv << 'EOF'
date,OT,Humidity,Wind Speed,Weather Type,Pressure,RH,VV,H,T,SLP,H_flag,T_flag,SLP_flag
2020-01-01 00:00:00,12.3,0.76,5.2,1,1013.25,76.0,10.0,100.0,12.3,1013.25,0,0,0
2020-01-01 01:00:00,12.1,0.77,5.1,1,1013.30,77.0,10.1,101.0,12.1,1013.30,0,0,0
2020-01-01 02:00:00,11.9,0.78,5.0,1,1013.35,78.0,10.2,102.0,11.9,1013.35,0,0,0
EOF

echo "✓ Sample weather dataset created"

# Create information file
echo ""
echo "Creating dataset information file..."
echo "-----------------------------------"

cat > dataset/README.md << 'EOF'
# FedTime Datasets

This directory contains the datasets used for FedTime experiments.

## Available Datasets

### 1. ETT (Electricity Transformer Temperature)
- **ETTh1.csv, ETTh2.csv**: Hourly data (17,420 data points each)
- **ETTm1.csv, ETTm2.csv**: 15-minute data (69,680 data points each)
- **Features**: 7 variables including target variable (OT) and 6 power load features
- **Time Range**: July 2016 - July 2018

### 2. Weather Dataset
- **weather.csv**: Sample weather dataset (for demo)
- **Features**: 21 meteorological indicators
- **Original**: Weather data from Max Planck Institute
- **Note**: Full dataset available at https://www.bgc-jena.mpg.de/wetter/

### 3. Traffic Dataset
- **traffic.csv**: Road occupancy rates (not included)
- **Features**: 862 sensors on San Francisco Bay area freeways
- **Original**: California Department of Transportation
- **Note**: Available at http://pems.dot.ca.gov

### 4. Electricity Dataset
- **electricity.csv**: Electricity consumption data (not included)
- **Features**: 321 customers (2012-2014)
- **Original**: UCI ML Repository
- **Note**: Available at https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

## Data Format

All datasets follow the format:
- First column: timestamp (date)
- Remaining columns: feature values
- Target variable typically named 'OT' (Oil Temperature)

## Usage

```python
# Load ETT dataset
import pandas as pd
data = pd.read_csv('dataset/ETT-small/ETTh1.csv')
print(data.head())
```

## Download Additional Datasets

For the complete datasets not included due to size:

1. **Weather**: Download from https://www.bgc-jena.mpg.de/wetter/
2. **Traffic**: Download from http://pems.dot.ca.gov
3. **Electricity**: Download from UCI ML Repository
4. **ACN**: Available at https://ev.caltech.edu/dataset

Place downloaded files in their respective subdirectories.
EOF

echo "✓ Dataset information file created"

# Check downloaded files
echo ""
echo "Download Summary:"
echo "=================="
echo "ETT Datasets:"
for file in dataset/ETT-small/*.csv; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "  ✓ $(basename $file) ($size)"
    fi
done

echo ""
echo "Other Datasets:"
if [ -f "dataset/weather/weather.csv" ]; then
    echo "  ✓ weather.csv (sample)"
else
    echo "  ✗ weather.csv (failed)"
fi

echo "  ⚠ traffic.csv (download manually from http://pems.dot.ca.gov)"
echo "  ⚠ electricity.csv (download manually from UCI ML Repository)"

echo ""
echo "=============================================="
echo "Dataset download completed!"
echo ""
echo "Next steps:"
echo "1. Run: python run_longExp.py --data ETTh1 --model FedTime"
echo "2. Or: python run_federated.py --data ETTh1 --model FedTime"
echo "=============================================="

# Make the script report success/failure
if [ -f "dataset/ETT-small/ETTh1.csv" ] && [ -f "dataset/ETT-small/ETTh2.csv" ]; then
    echo "✓ Essential datasets downloaded successfully"
    exit 0
else
    echo "✗ Some essential datasets failed to download"
    exit 1
fi
