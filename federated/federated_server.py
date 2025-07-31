import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Any


class FederatedServer:
    """
    Federated Learning Server for FedTime
    """
    
    def __init__(self, model, aggregator, device):
        self.model = model.to(device)
        self.aggregator = aggregator
        self.device = device
        self.round_num = 0
        
        # Store initial global parameters
        self.global_parameters = self.get_global_parameters()
        
        print(f"Federated server initialized with {self._count_parameters()} parameters")
    
    def get_global_parameters(self):
        """
        Get current global model parameters
        
        Returns:
            Dictionary of global model parameters
        """
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_global_parameters(self, parameters):
        """
        Set global model parameters
        
        Args:
            parameters: Dictionary of model parameters
        """
        for name, param in self.model.named_parameters():
            if name in parameters:
                param.data.copy_(parameters[name])
    
    def aggregate_updates(self, client_updates: List[Dict], client_weights: List[float]):
        """
        Aggregate client updates using the specified aggregation method
        
        Args:
            client_updates: List of client parameter dictionaries
            client_weights: List of weights for each client
            
        Returns:
            Aggregated global parameters
        """
        if not client_updates:
            return self.global_parameters
        
        # Normalize weights
        total_weight = sum(client_weights)
        if total_weight > 0:
            client_weights = [w / total_weight for w in client_weights]
        else:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)
        
        # Perform aggregation
        aggregated_params = self.aggregator.aggregate(
            client_updates, 
            client_weights, 
            self.global_parameters
        )
        
        # Update global model
        self.set_global_parameters(aggregated_params)
        self.global_parameters = aggregated_params
        self.round_num += 1
        
        return aggregated_params
    
    def evaluate_global_model(self, test_loader, criterion=None):
        """
        Evaluate global model on test data
        
        Args:
            test_loader: Test data loader
            criterion: Loss criterion (default: MSE)
            
        Returns:
            Test loss and metrics
        """
        if criterion is None:
            criterion = nn.MSELoss()
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = self.model(batch_x)
                pred = outputs[:, -batch_y.shape[1]:, :]  # Adjust based on pred_len
                true = batch_y
                
                loss = criterion(pred, true)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def _count_parameters(self):
        """Count total number of model parameters"""
        return sum(p.numel() for p in self.model.parameters())
    
    def get_model_size_bytes(self):
        """Calculate model size in bytes"""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size
    
    def save_global_model(self, path):
        """Save global model to disk"""
        torch.save(self.model.state_dict(), path)
        print(f"Global model saved to {path}")
    
    def load_global_model(self, path):
        """Load global model from disk"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.global_parameters = self.get_global_parameters()
        print(f"Global model loaded from {path}")


class ClusterServer(FederatedServer):
    """
    Federated Server with clustering support
    """
    
    def __init__(self, model, aggregator, device, num_clusters):
        super().__init__(model, aggregator, device)
        self.num_clusters = num_clusters
        self.cluster_models = {}
        self.cluster_parameters = {}
        
        # Initialize cluster-specific models
        for cluster_id in range(num_clusters):
            cluster_model = self._create_model_copy()
            self.cluster_models[cluster_id] = cluster_model
            self.cluster_parameters[cluster_id] = {
                name: param.data.clone() 
                for name, param in cluster_model.named_parameters()
            }
    
    def _create_model_copy(self):
        """Create a copy of the global model"""
        # This is a simplified implementation
        # In practice, you'd want to properly copy the model architecture
        import copy
        return copy.deepcopy(self.model)
    
    def aggregate_cluster_updates(self, cluster_id, client_updates, client_weights):
        """
        Aggregate updates for a specific cluster
        
        Args:
            cluster_id: ID of the cluster
            client_updates: List of client parameter dictionaries for this cluster
            client_weights: List of weights for each client in this cluster
            
        Returns:
            Aggregated cluster parameters
        """
        if cluster_id not in self.cluster_models:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        # Normalize weights
        total_weight = sum(client_weights)
        if total_weight > 0:
            client_weights = [w / total_weight for w in client_weights]
        else:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)
        
        # Aggregate updates for this cluster
        cluster_params = self.aggregator.aggregate(
            client_updates,
            client_weights,
            self.cluster_parameters[cluster_id]
        )
        
        # Update cluster model
        for name, param in self.cluster_models[cluster_id].named_parameters():
            if name in cluster_params:
                param.data.copy_(cluster_params[name])
        
        self.cluster_parameters[cluster_id] = cluster_params
        return cluster_params
    
    def get_cluster_parameters(self, cluster_id):
        """Get parameters for a specific cluster"""
        if cluster_id not in self.cluster_parameters:
            raise ValueError(f"Cluster {cluster_id} not found")
        return self.cluster_parameters[cluster_id]
    
    def aggregate_clusters(self, cluster_weights=None):
        """
        Aggregate all cluster models into a global model
        
        Args:
            cluster_weights: Optional weights for each cluster
        """
        if cluster_weights is None:
            cluster_weights = [1.0 / self.num_clusters] * self.num_clusters
        
        # Normalize cluster weights
        total_weight = sum(cluster_weights)
        if total_weight > 0:
            cluster_weights = [w / total_weight for w in cluster_weights]
        
        # Aggregate cluster parameters
        cluster_params_list = [self.cluster_parameters[i] for i in range(self.num_clusters)]
        
        global_params = self.aggregator.aggregate(
            cluster_params_list,
            cluster_weights,
            self.global_parameters
        )
        
        # Update global model
        self.set_global_parameters(global_params)
        self.global_parameters = global_params
        
        return global_params
    
    def evaluate_cluster_model(self, cluster_id, test_loader, criterion=None):
        """
        Evaluate a specific cluster model
        
        Args:
            cluster_id: ID of the cluster to evaluate
            test_loader: Test data loader
            criterion: Loss criterion
            
        Returns:
            Test loss for the cluster model
        """
        if cluster_id not in self.cluster_models:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        if criterion is None:
            criterion = nn.MSELoss()
        
        model = self.cluster_models[cluster_id]
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = model(batch_x)
                pred = outputs[:, -batch_y.shape[1]:, :]
                true = batch_y
                
                loss = criterion(pred, true)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
