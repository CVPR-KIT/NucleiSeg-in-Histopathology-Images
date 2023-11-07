import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class StratifiedClusterDataset(Dataset):
    def __init__(self, features, labels, k=1):
        """
        features: numpy array of features.
        labels: cluster labels corresponding to each feature.
        k: number of samples to take from center and boundary of each cluster.
        """
        self.features = features
        self.labels = labels
        self.k = k
        self.transformed_features = features  # Assuming features are already transformed
        self.selected_indices = self._select_indices()

    def _select_indices(self):
        unique_clusters = np.unique(self.labels)
        selected_indices = []

        for cluster in unique_clusters:
            if cluster != -1:  # Exclude the noise cluster
                cluster_indices = np.where(self.labels == cluster)[0]
                center, boundary = self.sample_from_cluster(cluster_indices, self.k)
                selected_indices.extend(center)
                selected_indices.extend(boundary)

        return np.array(selected_indices)

    def sample_from_cluster(self, cluster_indices, k=1):
        centroid = np.mean(self.transformed_features[cluster_indices], axis=0)
        distances = np.linalg.norm(self.transformed_features[cluster_indices] - centroid, axis=1)

        center_indices = cluster_indices[np.argsort(distances)[:k]]
        boundary_indices = cluster_indices[np.argsort(distances)[-k:]]

        return center_indices, boundary_indices

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        selected_idx = self.selected_indices[idx]
        return self.features[selected_idx], self.labels[selected_idx] 

# Usage with DataLoader
