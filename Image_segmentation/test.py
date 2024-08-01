import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from model import UNET
from utils import load_checkpoint, get_loaders

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1  # Test batch size
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
TEST_IMG_DIR = "data/test_images/"
TEST_MASK_DIR = "data/test_masks/"
CHECKPOINT_FILE = "my_checkpoint.pth.tar"

def visualize_prediction(model, loader, device):
    model.eval()
    with torch.no_grad():
        for idx, (data, targets) in enumerate(loader):
            data = data.to(device)
            targets = targets.to(device)

            predictions = torch.sigmoid(model(data))
            predictions = (predictions > 0.5).float()

            data = data.cpu().squeeze(0).permute(1, 2, 0).numpy()
            targets = targets.cpu().squeeze(0).numpy()
            predictions = predictions.cpu().squeeze(0).numpy()

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(data)

            plt.subplot(1, 3, 2)
            plt.title("Ground Truth")
            plt.imshow(targets, cmap='gray')

            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask")
            plt.imshow(predictions, cmap='gray')

            plt.show()

            # Display only one set of images
            break

def main():
    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load(CHECKPOINT_FILE), model)

    _, test_loader = get_loaders(
        None,  # Train image directory (not needed for testing)
        None,  # Train mask directory (not needed for testing)
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        None,  # Train transformations (not needed for testing)
        test_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    visualize_prediction(model, test_loader, DEVICE)

if __name__ == "__main__":
    main()
