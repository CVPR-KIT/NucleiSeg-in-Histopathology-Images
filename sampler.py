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


# Load images from path
def load_images(path):
    images = []
    print(f"loading Images from path: {path}")
    for filename in tqdm(natsorted(os.listdir(path))):
        if filename.endswith("_label.png"):
            continue
        img = cv2.imread(os.path.join(path,filename))
        if img is not None:
            images.append(img)
    return images


def load_model(modelType):
    if modelType == "small":
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    elif modelType == "large":
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    elif modelType == "giga":
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    else:
        print("Invalid Sampler model type")
        return None
    dino_model = dino_model.cuda()
    return dino_model


#process image patches
def get_features(image_patches, modelType="small"):
    features = []
    dino_model = load_model(modelType)

    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    

    for img in tqdm(image_patches):
        img_tensor = transform(img).unsqueeze(0)
        img_tensor = img_tensor.to('cuda')
        with torch.no_grad():
            feature = dino_model(img_tensor)
            feature = feature.cpu()
            features.append(feature.squeeze().numpy())

    return np.array(features)

def apply_tsne(features):
    tsne = TSNE(n_components=2, perplexity=40, learning_rate=200, random_state=42)
    image_patches_tsne = tsne.fit_transform(features)
    plt.scatter(image_patches_tsne[:, 0], image_patches_tsne[:, 1])
    plt.title('t-SNE Visualization')
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.imsave("tsne.png", image_patches_tsne)
    return image_patches_tsne

def apply_dbscan(tsnePatch, eps = 5, min_samples = 5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(tsnePatch)
    print("Unique clusters:", np.unique(clusters))  # You should see more than just -1
    return clusters

def sample_from_cluster(image_patches_tsne, cluster_indices, k=1):
    centroid = np.mean(image_patches_tsne[cluster_indices], axis=0)
    distances = np.linalg.norm(image_patches_tsne[cluster_indices] - centroid, axis=1)

    center_indices = cluster_indices[np.argsort(distances)[:k]]
    boundary_indices = cluster_indices[np.argsort(distances)[-k:]]

    return center_indices, boundary_indices


def sampleImages(config, batch_size=16):
    imgPaths = config["trainDataset"]
    image_patches = load_images(imgPaths)
    features = get_features(image_patches)
    image_patches_tsne = apply_tsne(features)
    clusters = apply_dbscan(image_patches_tsne)


    # Get unique clusters excluding noise
    valid_clusters = [c for c in np.unique(clusters) if c != -1]

    # Determine how many samples to take from each cluster
    samples_per_cluster = batch_size // (2 * len(valid_clusters))
    if samples_per_cluster == 0:
        samples_per_cluster = 1

    batch_indices = []
    for cluster in valid_clusters:
        cluster_indices = np.where(clusters == cluster)[0]
        center_indices, boundary_indices = sample_from_cluster(image_patches_tsne=image_patches_tsne, cluster_indices=cluster_indices, k=samples_per_cluster)

        batch_indices.extend(center_indices)
        batch_indices.extend(boundary_indices)

    # If the batch is not full, fill it with additional samples
    while len(batch_indices) < 16:
        additional_cluster = np.random.choice(valid_clusters)
        additional_indices = np.where(clusters == additional_cluster)[0]
        center_indices, _ = sample_from_cluster(additional_indices, k=1)
        batch_indices.extend(center_indices)

    # return batch indices
    return batch_indices[:batch_size]

if __name__ == '__main__':
    config = readConfig("config.sys")
    batch_indices = sampleImages(config)
    print(batch_indices)
