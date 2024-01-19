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
    def __init__(self, images, dino_model, config, mode="train", dbscan_eps=2, training_phase='high-density'):
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
        self.batchVisualization = config["batchVisualization"]
        
        self.dbscan_eps = dbscan_eps
        #self.plotDir = config["expt_dir"]
        # Perform feature extraction, t-SNE, and DBSCAN here 

        self.training_phase = training_phase

        self.count_insufficientBatch = 0


        self.image_patches = images
        if config["reUseFeatures"]:
            print("Loading Features")
            self.features = np.load("Outputs/Features/"+self.mode+"-features.npy")
        else:
            print("Calculating Features")
            self.features = self.get_features()
            print("Shape of Features: ", self.features.shape)
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

    def plot_batches(self, all_indices, total_batches):

        #plotted indices
        plotted_indices = set()

        for batch_num in range(total_batches):
            plt.figure(figsize=(8, 8))
            
            # Plot all points in a light grey color as a background
            plt.scatter(self.image_patches_tsne[:, 0], self.image_patches_tsne[:, 1], color='lightgrey', alpha=0.5)
            
            # Highlight the selected images for this batch
            selected_image_indexes = all_indices[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
            selected_tsne = self.image_patches_tsne[selected_image_indexes]
            old_tsne = self.image_patches_tsne[list(plotted_indices)]
            plotted_indices.update(selected_image_indexes)
    
            if old_tsne is not None:
                plt.scatter(old_tsne[:, 0], old_tsne[:, 1], color='green', alpha=0.6)  # Previously selected points in blue
            plt.scatter(selected_tsne[:, 0], selected_tsne[:, 1], color='red', alpha=0.6)  # Selected points in red

            plt.title(f't-SNE visualization of images for batch {batch_num + 1}')
            plt.xlabel('t-SNE component 1')
            plt.ylabel('t-SNE component 2')
            
            # Save the plot with the batch number
            plt.savefig(f"Outputs/Batch_Plots/tsne_batch_{batch_num + 1}.png")
            plt.close()


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

        # Plot the batches
        if self.batchVisualization:
            print("\nPlotting Batches for visualization")
            createDir(["Outputs/Batch_Plots/"])
            print("Total Batches: ", total_batches)
            self.plot_batches(all_indices, total_batches)

        # Yield each index at a time
        return iter(all_indices)



    def __len__(self):
        if self.debug:
            return len(self.image_patches) // self.debugDilution
        else:
            num_images = len(self.image_patches)
            return 820
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

    def save_batches(self, filename):
        all_batches = []
        total_batches = self.__len__()
        f_writer = open(filename, 'w')

        # Generate all batches
        for _ in range(total_batches):
            batch_indices = self.sampleImages()
            all_batches.append(batch_indices)
            f_writer.write(str(batch_indices) + '\n')
            if not len(batch_indices):
                break 




    def sample_from_cluster(self, cluster_indices, k=1, used_indices=None):
        if used_indices is None:
            used_indices = []

        # Extract the t-SNE coordinates for the current cluster
        cluster_tsne = self.image_patches_tsne[cluster_indices]

        # Calculate the centroid of the current cluster
        centroid = np.mean(cluster_tsne, axis=0)
        # Calculate the distances from each point in the cluster to the centroid
        distances = np.linalg.norm(cluster_tsne - centroid, axis=1)

        # Determine the maximum distance as the "radius" of the cluster
        max_distance = np.max(distances)
        # Set the threshold as a fraction of the maximum distance
        threshold_distance = max_distance / 2

        # Classify points as central or boundary
        central_indices = [cluster_indices[i] for i in range(len(cluster_indices)) if distances[i] <= threshold_distance and cluster_indices[i] not in used_indices]
        boundary_indices = [cluster_indices[i] for i in range(len(cluster_indices)) if distances[i] > threshold_distance and cluster_indices[i] not in used_indices]

        return central_indices[:k], boundary_indices[:k]
    

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


    def sampleImages(self):
        valid_clusters = [c for c in np.unique(self.clusters) if c >= 0]
        np.random.shuffle(valid_clusters)

        batch_indices = []
        loopCount = 0
        while len(batch_indices) < self.batch_size:
            new_indices = []
        
            for cluster in valid_clusters:
                if len(batch_indices) >= self.batch_size:
                    break

                center_indices, boundary_indices = self.sample_from_cluster(cluster_indices=np.where(self.clusters == cluster)[0], k=1, used_indices=self.all_indices)

                # Select indices based on the current training phase
                if self.training_phase == 'high-density':
                    new_indices = [idx for idx in center_indices if idx not in self.all_indices]
                else:  # low-density phase
                    new_indices = [idx for idx in boundary_indices if idx not in self.all_indices]

                batch_indices.extend(new_indices)
                self.all_indices.update(new_indices)

            if len(new_indices) == 0:
                loopCount += 1

            if loopCount > 10:
                # Handle the case where new indices are not found
                # Repeat some of the already selected indices to fill the batch
                remaining_slots = self.batch_size - len(batch_indices)
                repeat_indices = batch_indices[:remaining_slots]
                batch_indices.extend(repeat_indices)
                break  # Exit the while loop as the batch is now full
        #print("Batch Indices: ", batch_indices)

        return batch_indices[:self.batch_size]