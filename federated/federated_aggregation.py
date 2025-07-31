import torch
import numpy as np
from typing import List, Dict, Any
from collections import OrderedDict


class FederatedAggregator:
    """
    Base class for federated aggregation methods
    """
    
    def __init__(self):
        pass
    
    def aggregate(self, client_updates: List[Dict], client_weights: List[float], 
                  global_params: Dict) -> Dict:
        """
        Aggregate client updates
        
        Args:
            client_updates: List of client parameter dictionaries
            client_weights: List of weights for each client
            global_params: Current global parameters
            
        Returns:
            Aggregated parameters
        """
        raise NotImplementedError


class FedAvg(FederatedAggregator):
    """
    Federated Averaging (FedAvg) aggregation method
    """
    
    def __init__(self):
        super().__init__()
        
    def aggregate(self, client_updates: List[Dict], client_weights: List[float], 
                  global_params: Dict) -> Dict:
        """
        Perform FedAvg aggregation
        """
        if not client_updates:
            return global_params
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Get parameter names from first client
        param_names = client_updates[0].keys()
        
        for param_name in param_names:
            # Weighted average of parameters
            weighted_sum = torch.zeros_like(client_updates[0][param_name])
            
            for client_params, weight in zip(client_updates, client_weights):
                if param_name in client_params:
                    weighted_sum += weight * client_params[param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params


class FedAdam(FederatedAggregator):
    """
    Federated Adam (FedAdam) aggregation method
    """
    
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # Initialize momentum and variance estimates
        self.m = {}  # First moment estimate
        self.v = {}  # Second moment estimate
        self.t = 0   # Time step
        
    def aggregate(self, client_updates: List[Dict], client_weights: List[float], 
                  global_params: Dict) -> Dict:
        """
        Perform FedAdam aggregation
        """
        if not client_updates:
            return global_params
        
        # Increment time step
        self.t += 1
        
        # Compute pseudo-gradient (difference from global model)
        pseudo_grad = {}
        param_names = client_updates[0].keys()
        
        for param_name in param_names:
            # Weighted average of client parameters
            weighted_sum = torch.zeros_like(client_updates[0][param_name])
            
            for client_params, weight in zip(client_updates, client_weights):
                if param_name in client_params:
                    weighted_sum += weight * client_params[param_name]
            
            # Pseudo-gradient is difference between weighted average and global params
            if param_name in global_params:
                pseudo_grad[param_name] = weighted_sum - global_params[param_name]
            else:
                pseudo_grad[param_name] = weighted_sum
        
        # Apply Adam update
        aggregated_params = {}
        
        for param_name in param_names:
            grad = pseudo_grad[param_name]
            
            # Initialize momentum estimates if needed
            if param_name not in self.m:
                self.m[param_name] = torch.zeros_like(grad)
                self.v[param_name] = torch.zeros_like(grad)
            
            # Update biased first moment estimate
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            if param_name in global_params:
                aggregated_params[param_name] = global_params[param_name] + \
                    self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            else:
                aggregated_params[param_name] = self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        
        return aggregated_params


class FedOpt(FederatedAggregator):
    """
    Generic Federated Optimization (FedOpt) with different optimizers
    """
    
    def __init__(self, optimizer_type='sgd', lr=0.01, **optimizer_kwargs):
        super().__init__()
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs
        
        # Initialize optimizer state
        self.optimizer_state = {}
        self.t = 0
        
    def aggregate(self, client_updates: List[Dict], client_weights: List[float], 
                  global_params: Dict) -> Dict:
        """
        Perform FedOpt aggregation with specified optimizer
        """
        if not client_updates:
            return global_params
        
        # Increment time step
        self.t += 1
        
        # Compute pseudo-gradient
        pseudo_grad = {}
        param_names = client_updates[0].keys()
        
        for param_name in param_names:
            # Weighted average of client parameters
            weighted_sum = torch.zeros_like(client_updates[0][param_name])
            
            for client_params, weight in zip(client_updates, client_weights):
                if param_name in client_params:
                    weighted_sum += weight * client_params[param_name]
            
            # Pseudo-gradient
            if param_name in global_params:
                pseudo_grad[param_name] = weighted_sum - global_params[param_name]
            else:
                pseudo_grad[param_name] = weighted_sum
        
        # Apply optimizer-specific update
        if self.optimizer_type == 'sgd':
            return self._sgd_update(pseudo_grad, global_params)
        elif self.optimizer_type == 'adam':
            return self._adam_update(pseudo_grad, global_params)
        elif self.optimizer_type == 'rmsprop':
            return self._rmsprop_update(pseudo_grad, global_params)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
    
    def _sgd_update(self, pseudo_grad, global_params):
        """SGD update"""
        aggregated_params = {}
        
        for param_name, grad in pseudo_grad.items():
            if param_name in global_params:
                aggregated_params[param_name] = global_params[param_name] + self.lr * grad
            else:
                aggregated_params[param_name] = self.lr * grad
        
        return aggregated_params
    
    def _adam_update(self, pseudo_grad, global_params):
        """Adam update (same as FedAdam)"""
        beta1 = self.optimizer_kwargs.get('beta1', 0.9)
        beta2 = self.optimizer_kwargs.get('beta2', 0.999)
        eps = self.optimizer_kwargs.get('eps', 1e-8)
        
        aggregated_params = {}
        
        for param_name, grad in pseudo_grad.items():
            # Initialize state if needed
            if param_name not in self.optimizer_state:
                self.optimizer_state[param_name] = {
                    'm': torch.zeros_like(grad),
                    'v': torch.zeros_like(grad)
                }
            
            state = self.optimizer_state[param_name]
            
            # Update biased first moment estimate
            state['m'] = beta1 * state['m'] + (1 - beta1) * grad
            
            # Update biased second raw moment estimate
            state['v'] = beta2 * state['v'] + (1 - beta2) * (grad ** 2)
            
            # Compute bias-corrected estimates
            m_hat = state['m'] / (1 - beta1 ** self.t)
            v_hat = state['v'] / (1 - beta2 ** self.t)
            
            # Update parameters
            if param_name in global_params:
                aggregated_params[param_name] = global_params[param_name] + \
                    self.lr * m_hat / (torch.sqrt(v_hat) + eps)
            else:
                aggregated_params[param_name] = self.lr * m_hat / (torch.sqrt(v_hat) + eps)
        
        return aggregated_params
    
    def _rmsprop_update(self, pseudo_grad, global_params):
        """RMSprop update"""
        alpha = self.optimizer_kwargs.get('alpha', 0.99)
        eps = self.optimizer_kwargs.get('eps', 1e-8)
        
        aggregated_params = {}
        
        for param_name, grad in pseudo_grad.items():
            # Initialize state if needed
            if param_name not in self.optimizer_state:
                self.optimizer_state[param_name] = {
                    'v': torch.zeros_like(grad)
                }
            
            state = self.optimizer_state[param_name]
            
            # Update exponential moving average of squared gradients
            state['v'] = alpha * state['v'] + (1 - alpha) * (grad ** 2)
            
            # Update parameters
            if param_name in global_params:
                aggregated_params[param_name] = global_params[param_name] + \
                    self.lr * grad / (torch.sqrt(state['v']) + eps)
            else:
                aggregated_params[param_name] = self.lr * grad / (torch.sqrt(state['v']) + eps)
        
        return aggregated_params


class FedProx(FederatedAggregator):
    """
    Federated Proximal (FedProx) aggregation method
    """
    
    def __init__(self, mu=0.01):
        super().__init__()
        self.mu = mu  # Proximal term coefficient
        
    def aggregate(self, client_updates: List[Dict], client_weights: List[float], 
                  global_params: Dict) -> Dict:
        """
        Perform FedProx aggregation (similar to FedAvg but with proximal term during training)
        Note: The proximal term is applied during local training, not aggregation
        """
        # FedProx uses the same aggregation as FedAvg
        fedavg = FedAvg()
        return fedavg.aggregate(client_updates, client_weights, global_params)


class ScaleAggregator(FederatedAggregator):
    """
    Aggregator that scales updates based on client performance or data size
    """
    
    def __init__(self, scaling_method='uniform'):
        super().__init__()
        self.scaling_method = scaling_method
        
    def aggregate(self, client_updates: List[Dict], client_weights: List[float], 
                  global_params: Dict, client_metrics: List[float] = None) -> Dict:
        """
        Aggregate with performance-based or data-size-based scaling
        
        Args:
            client_updates: List of client parameter dictionaries
            client_weights: List of weights for each client
            global_params: Current global parameters
            client_metrics: Optional list of client performance metrics
        """
        if self.scaling_method == 'performance' and client_metrics is not None:
            # Scale weights based on inverse of loss (better performance = higher weight)
            scaled_weights = [1.0 / (metric + 1e-8) for metric in client_metrics]
            # Normalize
            total_weight = sum(scaled_weights)
            scaled_weights = [w / total_weight for w in scaled_weights]
        else:
            scaled_weights = client_weights
            
        # Use FedAvg with scaled weights
        fedavg = FedAvg()
        return fedavg.aggregate(client_updates, scaled_weights, global_params)
