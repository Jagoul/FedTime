import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import json
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from federated.server import FederatedServer
from federated.client import FederatedClient
from federated.aggregation import FedAvg, FedAdam, FedOpt
from federated.clustering import KMeansClustering
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings('ignore')


class Exp_Federated(Exp_Basic):
    """
    Federated Learning Experiment for FedTime
    """
    
    def __init__(self, args):
        super(Exp_Federated, self).__init__(args)
        self.args = args
        
        # Federated learning parameters
        self.num_clients = args.num_clients
        self.num_rounds = args.num_rounds
        self.local_epochs = args.local_epochs
        self.client_fraction = args.client_fraction
        
        # Initialize clustering if enabled
        if args.use_clustering:
            self.clustering = KMeansClustering(
                num_clusters=args.num_clusters,
                features=args.clustering_features
            )
        else:
            self.clustering = None
            
        # Initialize aggregation method
        if args.aggregation_method == 'fedavg':
            self.aggregator = FedAvg()
        elif args.aggregation_method == 'fedadam':
            self.aggregator = FedAdam(lr=args.server_lr)
        elif args.aggregation_method == 'fedopt':
            self.aggregator = FedOpt(lr=args.server_lr)
        else:
            raise ValueError(f"Unknown aggregation method: {args.aggregation_method}")
            
        # Communication overhead tracking
        self.communication_overhead = {
            'total_bytes': 0,
            'total_messages': 0,
            'round_bytes': [],
            'round_messages': []
        }
        
    def _build_federated_data(self):
        """
        Build federated data splits for clients
        """
        # Get full dataset
        train_data, train_loader = data_provider(self.args, flag='train')
        vali_data, vali_loader = data_provider(self.args, flag='val')
        test_data, test_loader = data_provider(self.args, flag='test')
        
        # Create federated splits
        client_data_splits = self._create_federated_splits(train_data, vali_data)
        
        # Create client dataloaders
        client_loaders = []
        for i in range(self.num_clients):
            client_train_loader = DataLoader(
                client_data_splits[i]['train'],
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                drop_last=True
            )
            client_vali_loader = DataLoader(
                client_data_splits[i]['val'],
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                drop_last=True
            )
            client_loaders.append({
                'train': client_train_loader,
                'val': client_vali_loader
            })
            
        return client_loaders, test_loader
    
    def _create_federated_splits(self, train_data, vali_data):
        """
        Create federated data splits for clients
        Implements non-IID data distribution simulation
        """
        # Simple implementation: split data sequentially
        # In practice, you might want more sophisticated non-IID splits
        
        train_size = len(train_data)
        vali_size = len(vali_data)
        
        train_per_client = train_size // self.num_clients
        vali_per_client = vali_size // self.num_clients
        
        client_data_splits = []
        
        for i in range(self.num_clients):
            train_start = i * train_per_client
            train_end = (i + 1) * train_per_client if i < self.num_clients - 1 else train_size
            
            vali_start = i * vali_per_client
            vali_end = (i + 1) * vali_per_client if i < self.num_clients - 1 else vali_size
            
            client_train_data = torch.utils.data.Subset(train_data, range(train_start, train_end))
            client_vali_data = torch.utils.data.Subset(vali_data, range(vali_start, vali_end))
            
            client_data_splits.append({
                'train': client_train_data,
                'val': client_vali_data
            })
            
        return client_data_splits
    
    def train(self, setting):
        """
        Federated training procedure
        """
        print(f"Starting federated training with {self.num_clients} clients for {self.num_rounds} rounds")
        
        # Prepare federated data
        client_loaders, test_loader = self._build_federated_data()
        
        # Initialize global model
        global_model = self._build_model()
        global_model.to(self.device)
        
        # Initialize federated server
        server = FederatedServer(
            model=global_model,
            aggregator=self.aggregator,
            device=self.device
        )
        
        # Initialize clients
        clients = []
        for i in range(self.num_clients):
            client = FederatedClient(
                client_id=i,
                model=self._build_model(),
                train_loader=client_loaders[i]['train'],
                val_loader=client_loaders[i]['val'],
                args=self.args,
                device=self.device
            )
            clients.append(client)
        
        # Perform clustering if enabled
        if self.clustering is not None:
            print("Performing client clustering...")
            client_features = self._extract_client_features(clients)
            cluster_assignments = self.clustering.fit_predict(client_features)
            
            # Group clients by cluster
            clusters = {}
            for client_id, cluster_id in enumerate(cluster_assignments):
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(client_id)
            
            print(f"Clients grouped into {len(clusters)} clusters: {clusters}")
        else:
            clusters = {0: list(range(self.num_clients))}
        
        # Training loop
        best_loss = float('inf')
        train_history = []
        
        for round_num in range(self.num_rounds):
            print(f"\n=== Federated Round {round_num + 1}/{self.num_rounds} ===")
            
            round_losses = []
            round_bytes = 0
            round_messages = 0
            
            # Process each cluster
            for cluster_id, client_ids in clusters.items():
                print(f"Processing cluster {cluster_id} with clients {client_ids}")
                
                # Select clients for this round
                num_selected = max(1, int(len(client_ids) * self.client_fraction))
                selected_clients = np.random.choice(client_ids, num_selected, replace=False)
                
                # Distribute global model to selected clients
                for client_id in selected_clients:
                    clients[client_id].set_model_parameters(server.get_global_parameters())
                    round_messages += 1
                    round_bytes += self._calculate_model_size(global_model)
                
                # Local training
                client_updates = []
                client_losses = []
                
                for client_id in selected_clients:
                    print(f"Training client {client_id}...")
                    client_loss = clients[client_id].local_train(self.local_epochs)
                    client_updates.append(clients[client_id].get_model_parameters())
                    client_losses.append(client_loss)
                    
                    round_messages += 1
                    round_bytes += self._calculate_model_size(global_model)
                
                # Aggregate updates for this cluster
                cluster_weights = [1.0 / len(selected_clients)] * len(selected_clients)
                cluster_global_params = server.aggregate_updates(client_updates, cluster_weights)
                
                round_losses.extend(client_losses)
            
            # Update communication overhead
            self.communication_overhead['round_bytes'].append(round_bytes)
            self.communication_overhead['round_messages'].append(round_messages)
            self.communication_overhead['total_bytes'] += round_bytes
            self.communication_overhead['total_messages'] += round_messages
            
            # Evaluate global model
            avg_loss = np.mean(round_losses)
            val_loss = self._evaluate_global_model(server.model, test_loader)
            
            train_history.append({
                'round': round_num + 1,
                'train_loss': avg_loss,
                'val_loss': val_loss
            })
            
            print(f"Round {round_num + 1} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Communication - Bytes: {round_bytes}, Messages: {round_messages}")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                self._save_model(server.model, setting)
        
        # Save training history and communication overhead
        self._save_training_results(train_history, setting)
        self._save_communication_overhead(setting)
        
        print(f"\nFederated training completed!")
        print(f"Best validation loss: {best_loss:.4f}")
        print(f"Total communication - Bytes: {self.communication_overhead['total_bytes']}, "
              f"Messages: {self.communication_overhead['total_messages']}")
    
    def _extract_client_features(self, clients):
        """
        Extract features from clients for clustering
        """
        client_features = []
        
        for client in clients:
            if self.args.clustering_features == 'data_stats':
                # Extract data statistics
                features = client.get_data_statistics()
            elif self.args.clustering_features == 'model_params':
                # Extract model parameter statistics
                features = client.get_model_statistics()
            else:
                # Mixed features
                data_features = client.get_data_statistics()
                model_features = client.get_model_statistics()
                features = np.concatenate([data_features, model_features])
            
            client_features.append(features)
        
        return np.array(client_features)
    
    def _evaluate_global_model(self, model, test_loader):
        """
        Evaluate global model on test data
        """
        model.eval()
        total_loss = 0.0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                outputs = model(batch_x)
                pred = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]
                
                loss = criterion(pred, true)
                total_loss += loss.item()
        
        return total_loss / len(test_loader)
    
    def _calculate_model_size(self, model):
        """
        Calculate model size in bytes for communication overhead
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size
    
    def _save_training_results(self, history, setting):
        """
        Save training history to file
        """
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        with open(os.path.join(folder_path, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def _save_communication_overhead(self, setting):
        """
        Save communication overhead results
        """
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        with open(os.path.join(folder_path, 'communication_overhead.json'), 'w') as f:
            json.dump(self.communication_overhead, f, indent=2)
    
    def _save_model(self, model, setting):
        """
        Save the best model
        """
        folder_path = './checkpoints/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        torch.save(model.state_dict(), os.path.join(folder_path, 'checkpoint.pth'))
    
    def test(self, setting, test=0):
        """
        Test the federated model
        """
        # Load test data
        test_data, test_loader = data_provider(self.args, flag='test')
        
        # Load trained model
        model = self._build_model()
        model_path = os.path.join('./checkpoints/' + setting + '/', 'checkpoint.pth')
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        else:
            print("No trained model found, using randomly initialized model")
        
        model.to(self.device)
        model.eval()
        
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('Testing...')
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x)
                else:
                    outputs = model(batch_x)

                pred = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # Result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return mae, mse, rmse, mape, mspe