# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to 128x128
    transforms.ToTensor()
])


# Custom Dataset class to load images without requiring class folders
class CelebADataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image  # No labels, since CelebA does not have direct class folders

# Function to create masked (damaged) images by adding black occlusions
def damage_image(image, mask_size=32):
    damaged = image.clone()
    _, h, w = image.shape
    x = np.random.randint(0, w - mask_size)
    y = np.random.randint(0, h - mask_size)
    damaged[:, y:y+mask_size, x:x+mask_size] = 0  # Black patch
    return damaged

# Define an Autoencoder Model for Image Restoration
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Match kernel_size=4
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize the model
model = Autoencoder()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Ensure correct device usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model Architecture
model = Autoencoder().to(device)

# Load Trained Weights
state_dict = torch.load("face_regeneration.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()  # Set model to evaluation mode

# Function to regenerate the image from the damaged image
# Function to regenerate the image from the damaged image
def regenerate1(test_image):

    with torch.no_grad():
        test_image = test_image.unsqueeze(0).to(device)  # Add batch dimension

        # Apply damage function and restore
        restored_image = model(test_image).cpu()

    # Convert tensor for display
    restored_image = restored_image.squeeze(0).permute(1, 2, 0).numpy()

    # Ensure values are in [0, 1] range
    restored_image = np.clip(restored_image, 0, 1)

    # Compute evaluation metrics
    mse_loss = np.mean((restored_image - test_image.cpu().numpy().squeeze(0).transpose(1, 2, 0)) ** 2)
    psnr_value = 10 * np.log10(1 / mse_loss) if mse_loss != 0 else float('inf')
    ssim_value = ssim(
        test_image.cpu().squeeze(0).permute(1, 2, 0).numpy(),
        restored_image,
        data_range=1.0,  # Explicitly define the data range for floating point images
        channel_axis=-1  # Explicitly define the channel axis
    )

    l1_loss = np.mean(np.abs(restored_image - test_image.cpu().numpy().squeeze(0).transpose(1, 2, 0)))
    
    evaluation_metrices_1 = [mse_loss,psnr_value,ssim_value,l1_loss]
    return restored_image, evaluation_metrices_1

def regenerate2(test_image):

    with torch.no_grad():
        test_image = test_image.unsqueeze(0).to(device)  # Add batch dimension

        restored_image = model(test_image).cpu()  # Restore image 

    # Convert tensor for display if needed
    def denormalize(tensor):
        return (tensor * 0.5) + 0.5  # Only if training used mean=0.5, std=0.5

    restored_image = denormalize(restored_image)

    restored_image2 = restored_image.squeeze(0)  # Removing the batch dimension

    # Ensure values are in [0, 1] range
    restored_image2 = torch.clamp(restored_image2, 0, 1)

    # Compute evaluation metrics
    mse_loss = torch.mean((restored_image2 - test_image.cpu().squeeze(0)) ** 2).item()
    psnr_value = 10 * np.log10(1 / mse_loss) if mse_loss != 0 else float('inf')
    ssim_value = ssim(
        test_image.cpu().squeeze(0).permute(1, 2, 0).numpy(),
        restored_image2.permute(1, 2, 0).numpy(),
        data_range=1.0,  # Explicitly define the data range for floating point images
        channel_axis=-1  # Explicitly define the channel axis
    )

    l1_loss = torch.mean(torch.abs(restored_image2 - test_image.cpu().squeeze(0))).item()

    evaluation_metrices_2 = [mse_loss,psnr_value,ssim_value,l1_loss]
    return restored_image2, evaluation_metrices_2