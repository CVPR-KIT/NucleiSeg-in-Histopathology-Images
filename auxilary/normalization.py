from torchvision import transforms
import torchstain
import cv2
import matplotlib.pyplot as plt
import numpy as np

def macenkoNormal(img, target):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])
    t_to_transform = T(img)
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    normalizer.fit(T(target))
    norm, H, E = normalizer.normalize(I=t_to_transform, stains=True)
    return norm.numpy().astype(np.uint8)


def reinhardNormal(img, target):
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])
    t_to_transform = T(img)
    normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
    normalizer.fit(T(target))
    norm = normalizer.normalize(I=t_to_transform)
    return norm


if __name__ == '__main__':
    img = cv2.imread("Dataset/test/1.png")
    target = cv2.imread("Dataset/sample/target.png")

    plt.imshow(macenkoNormal(img, target))