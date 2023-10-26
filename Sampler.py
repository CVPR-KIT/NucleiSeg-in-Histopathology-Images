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

class DinoPoweredSampler(Sampler):
    def __init__(self, images, dino_model, config, mode="train", pca_dim=0.5, dbscan_eps=5):
        '''
        Args:
            images: A list of image patches
            dino_model: The DINO model
            config: The config dictionary
            mode: The mode of the sampler. Can be "train", "val", "test" or "debug"
            pca_dim: The number of dimensions to reduce the features to using PCA. Can be 0.5 to keep 50% of the dimensions or an integer to keep that many dimensions.
            dbscan_eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster. 
                        This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
        '''
        self.dino_model = dino_model
        self.mode = mode
        self.batch_size = config["batch_size"]
        self.debug = config["debug"]
        self.debugDilution = config["debugDilution"]
        self.pca_dim = pca_dim
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
        #print("Applying t-SNE")
        #self.image_patches_tsne = self.apply_tsne()
        # Dim reduction using PCA
        pca = PCA(n_components=self.pca_dim)
        pca.fit(self.features)
        self.transformed_features = pca.transform(self.scaled_features)

        print("Applying DBSCAN")
        self.clusters = self.apply_dbscan()
        #np.save('Outputs/Features/image_clusters.npy', self.clusters)
        #print("Applying t-SNE")
        self.image_patches_tsne = self.apply_tsne()

        #np.save('Outputs/Features/image_patches_tsne.npy', self.image_patches_tsne)

        print("Sampling Initialization Complete") 


    def __iter__(self):
        # You can reuse your existing sampleImages function here
        # but modify it to return indices instead of image patches.
        #batch_indices = self.sampleImages()
        #return iter(batch_indices)
        all_indices = []
        num_images = len(self.image_patches)
        print("\nMaking Batches")
        
        for _ in tqdm(range(num_images // self.batch_size)):
            batch_indices = self.sampleImages()
            all_indices.extend(batch_indices)
        
        # Handle remaining images if any
        remaining_images = num_images % self.batch_size
        if remaining_images > 0:
            remaining_indices = self.sampleImages()[:remaining_images]
            all_indices.extend(remaining_indices)
        
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

    def sample_from_cluster(self, cluster_indices, k=1):
        centroid = np.mean(self.transformed_features[cluster_indices], axis=0)
        distances = np.linalg.norm(self.transformed_features[cluster_indices] - centroid, axis=1)

        center_indices = cluster_indices[np.argsort(distances)[:k]]
        boundary_indices = cluster_indices[np.argsort(distances)[-k:]]

        return center_indices, boundary_indices

    def apply_tsne(self):
        tsne = TSNE(n_components=2, perplexity=40, learning_rate=200, random_state=42)
        image_patches_tsne = tsne.fit_transform(self.transformed_features)
        # Plot the results
        plt.figure(1)
        plt.scatter(image_patches_tsne[:, 0], image_patches_tsne[:, 1], c=self.clusters)
        plt.colorbar()
        plt.title(f"t-SNE Visualization, DBSNE c - {len(np.unique(self.clusters))}")
        plt.xlabel('Y')
        plt.ylabel('X')
        #plt.imsave(self.plotDir+self.mode+"-tsne.png", image_patches_tsne)
        createDir(["Outputs/Plots/"])
        if self.mode == "debug":
            plt.savefig(f"Outputs/Plots/{self.mode}-tsne-{self.pca_dim}-{self.dbscan_eps}.png")
        else:
            plt.savefig("Outputs/Plots/"+self.mode+"-tsne.png")
        plt.clf()
        return image_patches_tsne

    def apply_dbscan(self, eps = 5, min_samples = 5, gen_plot = False):
        eps = self.dbscan_eps
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(self.transformed_features)
        print("Unique clusters:", np.unique(clusters))  # You should see more than just -1

        if not gen_plot:
            return clusters
        
        # Plot the results
        plt.figure(figsize=(10, 10))

        # Scatter plot for each uniquely labeled cluster
        unique_clusters = np.unique(clusters)
        for cluster in unique_clusters:
            x = self.transformed_features[clusters == cluster][:, 0]
            y = self.transformed_features[clusters == cluster][:, 1]
            plt.scatter(x, y, label=f"Cluster {cluster}")

        plt.title("DBSCAN Clustering")
        plt.xlabel("1st component")
        plt.ylabel("2nd component")
        plt.legend()
        createDir(["Outputs/Plots/"])
        
        if self.mode == "debug":
            plt.savefig(f"Outputs/Plots/{self.mode}-dbscan-{self.pca_dim}-{self.dbscan_eps}.png")
        else:
            plt.savefig("Outputs/Plots/"+self.mode+"-dbscan.png")
            
        return clusters

    def sampleImages(self):
        # Get unique clusters excluding noise
        valid_clusters = [c for c in np.unique(self.clusters)]

        # Determine how many samples to take from each cluster
        samples_per_cluster = self.batch_size // (2 * len(valid_clusters))
        if samples_per_cluster == 0:
            samples_per_cluster = 1

        batch_indices = []
        for cluster in valid_clusters:
            cluster_indices = np.where(self.clusters == cluster)[0]
            center_indices, boundary_indices = self.sample_from_cluster(cluster_indices=cluster_indices, k=samples_per_cluster)

            batch_indices.extend(center_indices)
            batch_indices.extend(boundary_indices)

    
        # If the batch is not full, fill it with additional samples
        while len(batch_indices) < 16:
            additional_cluster = np.random.choice(valid_clusters)
            additional_indices = np.where(self.clusters == additional_cluster)[0]
            center_indices, _ = self.sample_from_cluster(additional_indices, k=1)
            batch_indices.extend(center_indices)

        # return batch indices
        return batch_indices[:self.batch_size]