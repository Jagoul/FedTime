import torch
import torch.nn as nn
import numpy as np
from torch import optim
from utils.tools import adjust_learning_rate
from utils.metrics import metric


class FederatedClient:
    """
    Federated Learning Client for FedTime
    """
    
    def __init__(self, client_id, model, train_loader, val_loader, args, device):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device
        
        # Initialize optimizer
        self.optimizer = self._get_optimizer()
        self.criterion = nn.MSELoss()
        
        # Privacy parameters
        self.use_dp = args.differential_privacy if hasattr(args, 'differential_privacy') else False
        if self.use_dp:
            self.dp_noise_multiplier = args.dp_noise_multiplier
            self.dp_max_grad_norm = args.dp_max_grad_norm
        
        # Communication parameters
        self.compression = args.compression if hasattr(args, 'compression') else 'none'
        self.compression_ratio = args.compression_ratio if hasattr(args, 'compression_ratio') else 0.1
        
        print(f"Initialized client {client_id} with {len(train_loader)} training batches")
    
    def _get_optimizer(self):
        """Initialize optimizer for local training"""
        if self.args.use_peft:
            # Only optimize PEFT parameters
            trainable_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    trainable_params.append(param)
            return optim.Adam(trainable_params, lr=self.args.learning_rate)
        else:
            return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
    
    def local_train(self, epochs):
        """
        Perform local training for specified epochs
        
        Args:
            epochs: Number of local training epochs
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                
                # Calculate loss
                pred = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]
                loss = self.criterion(pred, true)
                
                # Backward pass
                loss.backward()
                
                # Apply differential privacy if enabled
                if self.use_dp:
                    self._apply_differential_privacy()
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss / len(self.train_loader)
            
            # Adjust learning rate
            adjust_learning_rate(self.optimizer, epoch + 1, self.args)
        
        avg_loss = total_loss / epochs
        print(f"Client {self.client_id} - Local training completed. Avg loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _apply_differential_privacy(self):
        """Apply differential privacy to gradients"""
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.dp_max_grad_norm)
        
        # Add noise to gradients
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0.0,
                    std=self.dp_noise_multiplier * self.dp_max_grad_norm,
                    size=param.grad.shape,
                    device=param.grad.device
                )
                param.grad += noise
    
    def get_model_parameters(self):
        """
        Get model parameters for federated aggregation
        
        Returns:
            Dictionary of model parameters
        """
        if self.args.use_peft:
            # Only return PEFT parameters
            params = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    params[name] = param.data.clone()
        else:
            params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        # Apply compression if enabled
        if self.compression != 'none':
            params = self._compress_parameters(params)
        
        return params
    
    def set_model_parameters(self, parameters):
        """
        Set model parameters from federated server
        
        Args:
            parameters: Dictionary of model parameters
        """
        # Decompress parameters if needed
        if self.compression != 'none':
            parameters = self._decompress_parameters(parameters)
        
        if self.args.use_peft:
            # Only update PEFT parameters
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in parameters:
                    param.data.copy_(parameters[name])
        else:
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.data.copy_(parameters[name])
    
    def _compress_parameters(self, parameters):
        """Apply gradient compression techniques"""
        if self.compression == 'topk':
            return self._topk_compression(parameters)
        elif self.compression == 'randomk':
            return self._randomk_compression(parameters)
        elif self.compression == 'quantization':
            return self._quantization_compression(parameters)
        else:
            return parameters
    
    def _decompress_parameters(self, parameters):
        """Decompress parameters"""
        # Implementation depends on compression method
        # For simplicity, we assume no decompression needed in this example
        return parameters
    
    def _topk_compression(self, parameters):
        """Top-k gradient compression"""
        compressed = {}
        for name, param in parameters.items():
            flat_param = param.flatten()
            k = max(1, int(len(flat_param) * self.compression_ratio))
            
            # Find top-k elements
            _, indices = torch.topk(torch.abs(flat_param), k)
            values = flat_param[indices]
            
            compressed[name] = {
                'values': values,
                'indices': indices,
                'shape': param.shape,
                'numel': param.numel()
            }
        
        return compressed
    
    def _randomk_compression(self, parameters):
        """Random-k gradient compression"""
        compressed = {}
        for name, param in parameters.items():
            flat_param = param.flatten()
            k = max(1, int(len(flat_param) * self.compression_ratio))
            
            # Randomly select k elements
            indices = torch.randperm(len(flat_param))[:k]
            values = flat_param[indices]
            
            compressed[name] = {
                'values': values,
                'indices': indices,
                'shape': param.shape,
                'numel': param.numel()
            }
        
        return compressed
    
    def _quantization_compression(self, parameters):
        """Quantization-based compression"""
        compressed = {}
        for name, param in parameters.items():
            # Simple 8-bit quantization
            param_min = param.min()
            param_max = param.max()
            
            # Quantize to 8-bit
            scale = (param_max - param_min) / 255.0
            quantized = torch.round((param - param_min) / scale).byte()
            
            compressed[name] = {
                'quantized': quantized,
                'min': param_min,
                'max': param_max,
                'shape': param.shape
            }
        
        return compressed
    
    def evaluate(self):
        """
        Evaluate model on validation data
        
        Returns:
            Validation loss and metrics
        """
        self.model.eval()
        total_loss = 0.0
        preds = []
        trues = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in self.val_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = self.model(batch_x)
                pred = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]
                
                loss = self.criterion(pred, true)
                total_loss += loss.item()
                
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        return avg_loss, mae, mse, rmse, mape, mspe
    
    def get_data_statistics(self):
        """
        Extract data statistics for clustering
        
        Returns:
            Array of data statistics features
        """
        # Calculate statistics from training data
        all_data = []
        for batch_x, batch_y, _, _ in self.train_loader:
            all_data.append(batch_x.numpy())
        
        data = np.concatenate(all_data, axis=0)  # [N, L, D]
        
        # Extract statistical features
        features = []
        
        # Mean and std per channel
        features.extend(np.mean(data, axis=(0, 1)))  # D features
        features.extend(np.std(data, axis=(0, 1)))   # D features
        
        # Temporal statistics
        features.append(np.mean(data))  # Overall mean
        features.append(np.std(data))   # Overall std
        features.append(np.min(data))   # Overall min
        features.append(np.max(data))   # Overall max
        
        # Skewness and kurtosis approximation
        from scipy import stats
        flat_data = data.flatten()
        features.append(stats.skew(flat_data))
        features.append(stats.kurtosis(flat_data))
        
        return np.array(features)
    
    def get_model_statistics(self):
        """
        Extract model parameter statistics for clustering
        
        Returns:
            Array of model statistics features
        """
        features = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_data = param.data.cpu().numpy()
                features.extend([
                    np.mean(param_data),
                    np.std(param_data),
                    np.min(param_data),
                    np.max(param_data)
                ])
        
        return np.array(features)
    
    def get_num_samples(self):
        """Get number of training samples for weighted aggregation"""
        return len(self.train_loader.dataset)
