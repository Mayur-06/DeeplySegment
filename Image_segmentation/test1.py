import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import os
from model import UNET
from utils import (
    load_checkpoint,
    get_loaders,
    save_predictions_as_imgs,
)

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
SAVE_DIR = "test_predictions/"

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

    save_predictions_as_imgs(
        test_loader, model, folder=SAVE_DIR, device=DEVICE
    )

if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    main()
