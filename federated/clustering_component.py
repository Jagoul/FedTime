import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List, Dict, Any, Tuple
import torch


class FederatedClustering:
    """
    Base class for federated clustering methods
    """
    
    def __init__(self, num_clusters: int):
        self.num_clusters = num_clusters
        self.cluster_assignments = None
        self.cluster_centers = None
        
    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Fit clustering model and predict cluster assignments
        
        Args:
            features: Feature matrix [n_clients, n_features]
            
        Returns:
            Cluster assignments for each client
        """
        raise NotImplementedError
    
    def get_cluster_info(self) -> Dict[int, List[int]]:
        """
        Get cluster information as dictionary
        
        Returns:
            Dict mapping cluster_id to list of client_ids
        """
        if self.cluster_assignments is None:
            return {}
        
        clusters = {}
        for client_id, cluster_id in enumerate(self.cluster_assignments):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(client_id)
        
        return clusters


class KMeansClustering(FederatedClustering):
    """
    K-means clustering for federated clients
    """
    
    def __init__(self, num_clusters: int, features: str = 'data_stats', 
                 normalize: bool = True, reduce_dim: bool = False, n_components: int = 10):
        super().__init__(num_clusters)
        self.features = features
        self.normalize = normalize
        self.reduce_dim = reduce_dim
        self.n_components = n_components
        
        # Initialize components
        self.scaler = StandardScaler() if normalize else None
        self.pca = PCA(n_components=n_components) if reduce_dim else None
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        
    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Perform K-means clustering on client features
        
        Args:
            features: Feature matrix [n_clients, n_features]
            
        Returns:
            Cluster assignments
        """
        # Handle edge cases
        if len(features) <= self.num_clusters:
            # If we have fewer clients than clusters, assign each to its own cluster
            return np.arange(len(features))
        
        processed_features = features.copy()
        
        # Normalize features
        if self.normalize and self.scaler is not None:
            processed_features = self.scaler.fit_transform(processed_features)
        
        # Dimensionality reduction
        if self.reduce_dim and self.pca is not None and processed_features.shape[1] > self.n_components:
            processed_features = self.pca.fit_transform(processed_features)
        
        # Perform clustering
        self.cluster_assignments = self.kmeans.fit_predict(processed_features)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        return self.cluster_assignments
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new clients
        
        Args:
            features: Feature matrix for new clients
            
        Returns:
            Predicted cluster assignments
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet")
        
        processed_features = features.copy()
        
        # Apply same preprocessing
        if self.normalize and self.scaler is not None:
            processed_features = self.scaler.transform(processed_features)
        
        if self.reduce_dim and self.pca is not None:
            processed_features = self.pca.transform(processed_features)
        
        return self.kmeans.predict(processed_features)


class HierarchicalClustering(FederatedClustering):
    """
    Hierarchical clustering for federated clients
    """
    
    def __init__(self, num_clusters: int, linkage: str = 'ward', 
                 normalize: bool = True):
        super().__init__(num_clusters)
        self.linkage = linkage
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        
    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Perform hierarchical clustering
        """
        from sklearn.cluster import AgglomerativeClustering
        
        if len(features) <= self.num_clusters:
            return np.arange(len(features))
        
        processed_features = features.copy()
        
        if self.normalize and self.scaler is not None:
            processed_features = self.scaler.fit_transform(processed_features)
        
        clustering = AgglomerativeClustering(
            n_clusters=self.num_clusters,
            linkage=self.linkage
        )
        
        self.cluster_assignments = clustering.fit_predict(processed_features)
        return self.cluster_assignments


class GaussianMixtureClustering(FederatedClustering):
    """
    Gaussian Mixture Model clustering for federated clients
    """
    
    def __init__(self, num_clusters: int, covariance_type: str = 'full',
                 normalize: bool = True):
        super().__init__(num_clusters)
        self.covariance_type = covariance_type
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        
    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Perform GMM clustering
        """
        from sklearn.mixture import GaussianMixture
        
        if len(features) <= self.num_clusters:
            return np.arange(len(features))
        
        processed_features = features.copy()
        
        if self.normalize and self.scaler is not None:
            processed_features = self.scaler.fit_transform(processed_features)
        
        gmm = GaussianMixture(
            n_components=self.num_clusters,
            covariance_type=self.covariance_type,
            random_state=42
        )
        
        gmm.fit(processed_features)
        self.cluster_assignments = gmm.predict(processed_features)
        return self.cluster_assignments


class PerformanceBasedClustering(FederatedClustering):
    """
    Clustering based on client performance metrics
    """
    
    def __init__(self, num_clusters: int, performance_threshold: float = 0.1):
        super().__init__(num_clusters)
        self.performance_threshold = performance_threshold
        
    def fit_predict(self, features: np.ndarray, 
                    performance_metrics: List[float] = None) -> np.ndarray:
        """
        Cluster clients based on performance metrics
        
        Args:
            features: Client features (may be used as additional info)
            performance_metrics: List of performance scores for each client
            
        Returns:
            Cluster assignments
        """
        if performance_metrics is None:
            # Fallback to regular K-means
            kmeans = KMeansClustering(self.num_clusters)
            return kmeans.fit_predict(features)
        
        # Sort clients by performance
        client_performance = list(enumerate(performance_metrics))
        client_performance.sort(key=lambda x: x[1])  # Sort by performance (lower is better)
        
        # Divide into clusters based on performance quantiles
        n_clients = len(client_performance)
        clients_per_cluster = n_clients // self.num_clusters
        
        self.cluster_assignments = np.zeros(n_clients, dtype=int)
        
        for i, (client_id, _) in enumerate(client_performance):
            cluster_id = min(i // clients_per_cluster, self.num_clusters - 1)
            self.cluster_assignments[client_id] = cluster_id
        
        return self.cluster_assignments


class DataDistributionClustering(FederatedClustering):
    """
    Clustering based on data distribution similarity
    """
    
    def __init__(self, num_clusters: int, similarity_metric: str = 'kl_divergence'):
        super().__init__(num_clusters)
        self.similarity_metric = similarity_metric
        
    def fit_predict(self, client_data_stats: List[Dict[str, Any]]) -> np.ndarray:
        """
        Cluster clients based on data distribution statistics
        
        Args:
            client_data_stats: List of data statistics for each client
            
        Returns:
            Cluster assignments
        """
        # Extract distribution features from data statistics
        features = self._extract_distribution_features(client_data_stats)
        
        # Use K-means on distribution features
        kmeans = KMeansClustering(self.num_clusters, normalize=True)
        return kmeans.fit_predict(features)
    
    def _extract_distribution_features(self, client_data_stats: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract distribution features from client data statistics
        """
        features = []
        
        for stats in client_data_stats:
            feature_vector = []
            
            # Extract statistical moments
            if 'mean' in stats:
                feature_vector.extend(np.atleast_1d(stats['mean']))
            if 'std' in stats:
                feature_vector.extend(np.atleast_1d(stats['std']))
            if 'skewness' in stats:
                feature_vector.extend(np.atleast_1d(stats['skewness']))
            if 'kurtosis' in stats:
                feature_vector.extend(np.atleast_1d(stats['kurtosis']))
            
            # Extract quantiles if available
            if 'quantiles' in stats:
                feature_vector.extend(stats['quantiles'])
            
            features.append(feature_vector)
        
        return np.array(features)


class AdaptiveClustering(FederatedClustering):
    """
    Adaptive clustering that changes cluster assignments over time
    """
    
    def __init__(self, num_clusters: int, adaptation_rate: float = 0.1,
                 min_rounds_between_updates: int = 10):
        super().__init__(num_clusters)
        self.adaptation_rate = adaptation_rate
        self.min_rounds_between_updates = min_rounds_between_updates
        self.last_update_round = 0
        self.round_counter = 0
        
    def should_update_clusters(self, current_round: int) -> bool:
        """
        Determine if clusters should be updated in current round
        """
        return (current_round - self.last_update_round) >= self.min_rounds_between_updates
    
    def fit_predict(self, features: np.ndarray, 
                    current_round: int = 0,
                    performance_history: List[List[float]] = None) -> np.ndarray:
        """
        Adaptive clustering based on recent performance
        """
        self.round_counter = current_round
        
        if not self.should_update_clusters(current_round) and self.cluster_assignments is not None:
            return self.cluster_assignments
        
        # Perform new clustering
        if performance_history is not None and len(performance_history) > 0:
            # Use recent performance for clustering
            recent_performance = performance_history[-1]  # Most recent round
            clustering = PerformanceBasedClustering(self.num_clusters)
            assignments = clustering.fit_predict(features, recent_performance)
        else:
            # Use feature-based clustering
            clustering = KMeansClustering(self.num_clusters)
            assignments = clustering.fit_predict(features)
        
        # Apply adaptation rate (gradual change)
        if self.cluster_assignments is not None:
            # Blend old and new assignments
            change_mask = np.random.random(len(assignments)) < self.adaptation_rate
            assignments = np.where(change_mask, assignments, self.cluster_assignments)
        
        self.cluster_assignments = assignments
        self.last_update_round = current_round
        
        return assignments


def create_clustering_method(method: str, num_clusters: int, **kwargs) -> FederatedClustering:
    """
    Factory function to create clustering methods
    
    Args:
        method: Clustering method name
        num_clusters: Number of clusters
        **kwargs: Additional arguments for the clustering method
        
    Returns:
        Clustering instance
    """
    if method.lower() == 'kmeans':
        return KMeansClustering(num_clusters, **kwargs)
    elif method.lower() == 'hierarchical':
        return HierarchicalClustering(num_clusters, **kwargs)
    elif method.lower() == 'gmm':
        return GaussianMixtureClustering(num_clusters, **kwargs)
    elif method.lower() == 'performance':
        return PerformanceBasedClustering(num_clusters, **kwargs)
    elif method.lower() == 'distribution':
        return DataDistributionClustering(num_clusters, **kwargs)
    elif method.lower() == 'adaptive':
        return AdaptiveClustering(num_clusters, **kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
