import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from auxilary.utils import *
from dataset import MonuSegDataSet, MonuSegValDataSet


def show_image(dataloader):

    for images, labels in dataloader:
        # Choose the first image in the batch
        image = images[0]

        # If the image is on the GPU, move it back to the CPU
        image = image.cpu().numpy()

        # If the image has 3 channels, transpose it to (height, width, channels)
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        # If the image is normalized, transform it back to the (0, 1) range
        mage = unnormalize_image(image)

        # Plot the image
        plt.imshow(image)
        plt.title(f"Label: {labels[0]}")
        plt.axis('off')
        plt.show()

        # Break after the first batch (remove this line if you want to see more)
        break




def main():
    # Load config
    config = readConfig()
    train_dataset = MonuSegDataSet(config["trainDataset"])
    train_data = DataLoader(train_dataset,batch_size=config["batch_size"],shuffle=True)

    val_dataset = MonuSegValDataSet(config["valDataset"])
    val_data = DataLoader(val_dataset,batch_size=1,num_workers=4)
    
    show_image(train_data)
