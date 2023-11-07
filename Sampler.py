from torch.utils.data import Sampler
import torch
from torchvision import transforms
import cv2
from tqdm import tqdm
from natsort import natsorted
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from auxilary.utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import itertools


class DinoPoweredSampler(Sampler):
    def __init__(self, images, dino_model, config, mode="train", dbscan_eps=2):
        '''
        Args:
            images: A list of image patches
            dino_model: The DINO model
            config: The config dictionary
            mode: The mode of the sampler. Can be "train", "val", "test" or "debug"
            dbscan_eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster. 
                        This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
        '''
        self.dino_model = dino_model
        self.mode = mode
        self.batch_size = config["batch_size"]
        self.debug = config["debug"]
        self.debugDilution = config["debugDilution"]
        
        self.dbscan_eps = dbscan_eps
        #self.plotDir = config["expt_dir"]
        # Perform feature extraction, t-SNE, and DBSCAN here 

        self.image_patches = images
        if config["reUseFeatures"]:
            print("Loading Features")
            self.features = np.load("Outputs/Features/"+self.mode+"-features.npy")
        else:
            print("Calculating Features")
            self.features = self.get_features()
            createDir(["Outputs/Features/"])
            np.save("Outputs/Features/"+self.mode+"-features.npy", self.features)


        #Scaling Features
        scaler = StandardScaler()
        self.scaled_features = scaler.fit_transform(self.features)

        normalizer = Normalizer(norm='l2')
        self.normalized_features = normalizer.fit_transform(self.scaled_features)

        self.image_patches_tsne = self.apply_tsne(plot=False)

        print("Applying DBSCAN")
        self.clusters = self.apply_dbscan()

        # plot the clusters
        _t = self.apply_tsne(plot=True)


        #np.save('Outputs/Features/image_clusters.npy', self.clusters)
        #print("Applying t-SNE")
        

        #np.save('Outputs/Features/image_patches_tsne.npy', self.image_patches_tsne)

        self.all_indices = set()

        print("Sampling Initialization Complete") 


    def __iter__0(self):
        # You can reuse your existing sampleImages function here
        # but modify it to return indices instead of image patches.
        #batch_indices = self.sampleImages()
        #return iter(batch_indices)
        num_images = len(self.image_patches)
        print("\nMaking Batches")
        
        for _ in tqdm(range(num_images // self.batch_size)):
            batch_indices = self.sampleImages()
            self.all_indices.update(batch_indices)
        
        # Handle remaining images if any
        remaining_images = num_images % self.batch_size
        if remaining_images > 0:
            remaining_indices = self.sampleImages()[:remaining_images]
            self.all_indices.update(remaining_indices)
        
        print("all_indices: ", len(self.all_indices))
        return iter(self.all_indices)
    

    def __iter__(self):
        # Reset all_indices at the beginning of each iteration to start fresh
        self.all_indices = set()

        # Initialize an empty list to store all indices for the epoch
        all_indices = []

        # Calculate the total number of batches needed
        total_batches = self.__len__()

        # Loop to generate all indices for the epoch
        for _ in range(total_batches):
            # Sample indices for a batch
            batch_indices = self.sampleImages()

            # Check if we have already included enough indices
            if len(all_indices) + len(batch_indices) > len(self.image_patches):
                # If adding the current batch_indices exceeds the number of images,
                # trim the batch_indices to fit the remaining number of images
                batch_indices = batch_indices[:len(self.image_patches) - len(all_indices)]

            # Extend the all_indices list with the new batch indices
            all_indices.extend(batch_indices)

        # Shuffle the indices to ensure random order of image access
        np.random.shuffle(all_indices)

        # Yield each index at a time
        return iter(all_indices)



    def __len__(self):
        if self.debug:
            return len(self.image_patches) // self.debugDilution
        else:
            num_images = len(self.image_patches)
            return (num_images // self.batch_size) + int(num_images % self.batch_size > 0)
    
    
    #process image patches
    def get_features(self):
        features = []

        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    

        for img in tqdm(self.image_patches):
            img_tensor = transform(img).unsqueeze(0)
            img_tensor = img_tensor.to('cuda')
            with torch.no_grad():
                feature = self.dino_model(img_tensor)
                feature = feature.cpu()
                features.append(feature.squeeze().numpy())

        return np.array(features)

    def sample_from_cluster0(self, cluster_indices, k=1):
        centroid = np.mean(self.image_patches_tsne[cluster_indices], axis=0)
        distances = np.linalg.norm(self.image_patches_tsne[cluster_indices] - centroid, axis=1)

        center_indices = cluster_indices[np.argsort(distances)[:k]]
        boundary_indices = cluster_indices[np.argsort(distances)[-k:]]

        return center_indices, boundary_indices
    
    def sample_from_cluster(self, cluster_indices, k=1, used_indices=None):
        if used_indices is None:
            used_indices = []
    
        # Compute centroid and distances as before
        centroid = np.mean(self.image_patches_tsne[cluster_indices], axis=0)
        distances = np.linalg.norm(self.image_patches_tsne[cluster_indices] - centroid, axis=1)

        # Find center indices, excluding used ones
        sorted_indices = np.argsort(distances)
        center_indices = [idx for idx in cluster_indices[sorted_indices] if idx not in used_indices][:k]

        # Find boundary indices, excluding used ones
        boundary_indices = [idx for idx in cluster_indices[sorted_indices[::-1]] if idx not in used_indices][:k]

        return center_indices, boundary_indices

    def apply_tsne(self, plot = False):
        if not plot:
            tsne = TSNE(n_components=2, perplexity=40, learning_rate=200, random_state=42)
            image_patches_tsne = tsne.fit_transform(self.normalized_features)
            return image_patches_tsne
        # Plot the results
        plt.figure(1)
        plt.scatter(self.image_patches_tsne[:, 0], self.image_patches_tsne[:, 1], c=self.clusters)
        plt.colorbar()
        plt.title(f"t-SNE Visualization, DBSNE c - {len(np.unique(self.clusters))}")
        plt.xlabel('Y')
        plt.ylabel('X')
        #plt.imsave(self.plotDir+self.mode+"-tsne.png", image_patches_tsne)
        createDir(["Outputs/Plots/"])
        if self.mode == "debug":
            plt.savefig(f"Outputs/Plots/{self.mode}-tsne-{self.dbscan_eps}.png")
        else:
            plt.savefig("Outputs/Plots/"+self.mode+"-tsne.png")
        plt.clf()
        return None

    def apply_dbscan(self, eps = 2, min_samples = 5, metrics='euclidean',gen_plot = False):
        eps = self.dbscan_eps
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metrics)
        clusters = dbscan.fit_predict(self.image_patches_tsne)
        print("Unique clusters:", np.unique(clusters))  # You should see more than just -1

        if not gen_plot:
            return clusters
        
        # Plot the results
        plt.figure(figsize=(10, 10))

        # Scatter plot for each uniquely labeled cluster
        unique_clusters = np.unique(clusters)
        for cluster in unique_clusters:
            x = self.image_patches_tsne[clusters == cluster][:, 0]
            y = self.image_patches_tsne[clusters == cluster][:, 1]
            plt.scatter(x, y, label=f"Cluster {cluster}")

        plt.title("DBSCAN Clustering")
        plt.xlabel("1st component")
        plt.ylabel("2nd component")
        plt.legend()
        createDir(["Outputs/Plots/"])
        
        if self.mode == "debug":
            plt.savefig(f"Outputs/Plots/{self.mode}-dbscan-{self.dbscan_eps}.png")
        else:
            plt.savefig("Outputs/Plots/"+self.mode+"-dbscan.png")
            
        return clusters

    def sampleImages0(self):
        # Get unique clusters excluding noise
        valid_clusters = [c for c in np.unique(self.clusters) if c >= 0]

        # Determine how many samples to take from each cluster
        samples_per_cluster = self.batch_size // (len(valid_clusters))
        if samples_per_cluster == 0:
            samples_per_cluster = 1

        batch_indices = []
        for cluster in valid_clusters:
            cluster_indices = np.where(self.clusters == cluster)[0]
            center_indices, boundary_indices = self.sample_from_cluster(cluster_indices=cluster_indices, k=samples_per_cluster)

            batch_indices.extend(center_indices)
            batch_indices.extend(boundary_indices)

    
        # If the batch is not full then fill it with additional samples
        while len(batch_indices) < self.batch_size:
            additional_cluster = np.random.choice(valid_clusters)
            additional_indices = np.where(self.clusters == additional_cluster)[0]
            center_indices, _ = self.sample_from_cluster(additional_indices, k=1)
            batch_indices.extend(center_indices)

        #np.random.shuffle(batch_indices)
        print(batch_indices[:self.batch_size])
        print(len(batch_indices))

        # return batch indices
        return batch_indices[:self.batch_size]
    
    
    def sampleImages(self):
        valid_clusters = [c for c in np.unique(self.clusters) if c >= 0]  # Excluding noise (-1)
        np.random.shuffle(valid_clusters)  # Shuffle clusters for randomness

        # Pre-select center and boundary indices for each cluster, exclude used indices
        preselected_indices = {cluster: self.sample_from_cluster(cluster_indices=np.where(self.clusters == cluster)[0], k=1)
                           for cluster in valid_clusters}

        batch_indices = []
        for cluster in valid_clusters:
            if len(batch_indices) >= self.batch_size:
                break  # Batch is full

            # Add new, unused indices to the batch
            center_indices, boundary_indices = preselected_indices[cluster]
            new_indices = [idx for idx in center_indices + boundary_indices if idx not in self.all_indices]

            # Append new indices and update all_indices
            batch_indices.extend(new_indices)
            self.all_indices.update(new_indices)  # Assuming all_indices is a set for O(1) lookups

        # Fill the rest of the batch if needed
        if len(batch_indices) < self.batch_size:
            all_unused_indices = [idx for idx in range(len(self.image_patches)) if idx not in self.all_indices]
            np.random.shuffle(all_unused_indices)
            batch_indices.extend(all_unused_indices[:self.batch_size - len(batch_indices)])

        #print(batch_indices[:self.batch_size])
        #print(len(batch_indices))

        return batch_indices[:self.batch_size]
