from auxilary.utils import readConfig
from auxilary.simplex import Simplex_CLASS as simplex
from torchvision import transforms
import cv2
import torchstain
import numpy as np
from tqdm import tqdm
import  argparse
from pathlib import Path
import random


def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='none', help='Path to the config file.')
    return parser.parse_args()

# Reinhard Normalization
def reinhardNormal(img, target):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])
    t_to_transform = T(img)
    normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
    normalizer.fit(T(target))
    norm = normalizer.normalize(I=t_to_transform)
    return norm.numpy().astype(np.uint8)


# Function to add noise to image
def noisy_image(img, alpha, random_state=None):
    if random_state < 0.5:
        return img

    # Generate noise
    simplexObj = simplex()
    img_size = (img.shape[0], img.shape[1])
    noise = simplexObj.rand_2d_octaves(img_size, 6, 0.6)
    # Convert image to float [0, 1]
    image_array = img.astype(np.float32) / 255
    # Normalize Noise to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())  
    # Blend noise with original image
    image_array = (1 - alpha) * image_array + alpha * noise[..., np.newaxis]
    # Convert back to uint8 [0, 255]
    image_array = (image_array * 255).astype(np.uint8)
    return image_array


# Function to distort image
def elastic_transform(image, alpha=100, sigma=10, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
     and https://github.com/rwightman/tensorflow-litterbox/blob/ddeeb3a6c7de64e5391050ffbb5948feca65ad3c/litterbox/fabric/image_processing_common.py#L220
    """
    if random_state < 0.5:
        return image

    shape_size = image.shape[:2]

    # Downscaling the random grid and then upsizing post filter
    # improves performance. Approx 3x for scale of 4, diminishing returns after.
    grid_scale = 4
    alpha //= grid_scale  # Does scaling these make sense? seems to provide
    sigma //= grid_scale  # more similar end result when scaling grid used.
    grid_shape = (shape_size[0]//grid_scale, shape_size[1]//grid_scale)

    blur_size = int(4 * sigma) | 1
    rand_x = cv2.GaussianBlur(
        (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
        ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    rand_y = cv2.GaussianBlur(
        (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
        ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    if grid_scale > 1:
        rand_x = cv2.resize(rand_x, shape_size[::-1])
        rand_y = cv2.resize(rand_y, shape_size[::-1])

    grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
    grid_x = (grid_x + rand_x).astype(np.float32)
    grid_y = (grid_y + rand_y).astype(np.float32)

    distorted_img = cv2.remap(image, grid_x, grid_y,
        borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR)

    return distorted_img

# function for gamma correction
def adjust_gamma(img, gamma):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # Apply the gamma correction using the lookup table
    return cv2.LUT(img, table)


def augmentImages(images, labels, outdir, augment_num_per_img, config):

    base_dir = outdir
    count = 0
    normalizationTargetpath  = config["targetImagePath"]
    target = cv2.imread(normalizationTargetpath)
    for i in tqdm(range(len(images))):
        
        elastic_random = np.random.uniform(size=augment_num_per_img)  # [0, 1]
        noise_random = np.random.uniform(size=augment_num_per_img)  # [0, 1]

        # normalize image
        images[i] = reinhardNormal(images[i], target)
        # correct label
        labels[i] = correctLabel(labels[i])

        
        for j in range(augment_num_per_img):
            modImage = images[i]
            modLabel = labels[i]
            # add noise 
            modImage = noisy_image(modImage, alpha=0.3, random_state=noise_random[j])
            # elastic transform
            if elastic_random[j] <= 0.8:
                modImage = elastic_transform(modImage, alpha=300, sigma=30, random_state=elastic_random[j])
                modLabel = elastic_transform(modLabel, alpha=300, sigma=30, random_state=elastic_random[j])

            # gamma adjustment
            gamma_values = [ 1, 1.5]

            for gamma_value in gamma_values:
                corrected_image = adjust_gamma(modImage, gamma_value)
                new_img_name = str(count) +".png"
                new_label_name = str(count) + "_label.png"
                cv2.imwrite(base_dir + new_img_name, corrected_image)
                cv2.imwrite(base_dir + new_label_name , modLabel)
                f.write(new_img_name+"\n"+new_label_name+"\n")
                count += 1
            
            # save image and label
            new_img_name = str(count) +".png"
            new_label_name = str(count) + "_label.png"
            cv2.imwrite(base_dir + new_img_name, modImage)
            cv2.imwrite(base_dir + new_label_name, modLabel)
            f.write(new_img_name+"\n"+new_label_name+"\n")
            count += 1

def correctLabel(label):
    # change all non black pixel to white
    for i in range(0,label.shape[0]):
        for j in range(0,label.shape[1]):
            if label[i][j] != 0:
                label[i][j] = 255
    return label * 255


def saveTestImages(images, labels, outdir, config):
    base_dir = outdir
    assert len(images) == len(labels)
    normalizationTargetPath = config["targetImagePath"]
    target = cv2.imread(normalizationTargetPath)
    for i in tqdm(range(len(images))):
        normalImage = reinhardNormal(images[i], target)
        labels[i] = correctLabel(labels[i])
        new_img_name = str(i) +".png"
        new_label_name = str(i) + "_label.png"
        cv2.imwrite(base_dir + new_img_name, normalImage)
        cv2.imwrite(base_dir + new_label_name, labels[i])

if __name__=='__main__':
    args = arg_init()

    if args.config == 'none':
        print("Please provide the path to the config file")
        exit()

    config = readConfig(args.config)

    log_dir = config["log"]
    f = open(log_dir + "logs-pre-training.txt", "a")
    print("Performing Photometric  Augmentations")
    f.write("Performing Photometric Augmentations\n")

    # GET CONFIGURATION DETAILS
    augment_num_per_img = config["augmentPerImage"]

    # CREATE DIRECTORIES
    trainDataset = config["trainDataset"]
    valDataset = config["valDataset"]
    testDataset = config["testDataset"]

    print("Checking and Creating Directories")
    path = Path(trainDataset)
    path.mkdir(parents=True, exist_ok=True)
    path = Path(valDataset)
    path.mkdir(parents=True, exist_ok=True)
    path = Path(testDataset)
    path.mkdir(parents=True, exist_ok=True)

    # Load images and labels
    imagePath = config["imagePath"]
    labelPath = config["labelPath"]
    print("Loading Images and Labels")
    images = np.load(imagePath)
    labels = np.load(labelPath)
    # since we are using segmentation only
    labels = labels[:,:,:,1]

    # Randomly shuffle images and labels
    print("Shuffling Images and Labels")
    random.seed(42)
    random.shuffle(images)
    random.seed(42)
    random.shuffle(labels)

    print("Splitting Images and Labels into Train, Val and Test")
    # split into train and val and test
    splitRatio = config["splitRatio"]
    split = int(len(images) * splitRatio)
    trainImages = images[:split]
    trainLabels = labels[:split]
    valImages = images[split:]
    valLabels = labels[split:]

    # create 1/3 of val images into test images
    splitRatio = 1/3
    split = int(len(valImages) * splitRatio)
    testImages = valImages[:split]
    testLabels = valLabels[:split]
    valImages = valImages[split:]
    valLabels = valLabels[split:]

    print("Saving Test Images")
    saveTestImages(testImages, testLabels, config["testDataset"], config)


    # perform augmentations
    print("Performing Augmentations on Train Images")
    augmentImages(trainImages, trainLabels, trainDataset, augment_num_per_img, config)
    print("Performing Augmentations on Val Images")
    augmentImages(valImages, valLabels, valDataset, augment_num_per_img, config)

    print("Augmentation Completed")
    f.write("Augmentation Completed\n")
    f.close()