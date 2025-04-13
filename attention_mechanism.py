# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Function to create masked (damaged) images
def damage_image(image, mask_size=32):
    damaged = image.clone()
    _, h, w = image.shape
    x = np.random.randint(0, w - mask_size)
    y = np.random.randint(0, h - mask_size)
    damaged[:, y:y+mask_size, x:x+mask_size] = 0  # Black patch
    return damaged

# Define a Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.shape
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        attention = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(attention, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

# Define an Autoencoder Model with Attention
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            SelfAttention(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            SelfAttention(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            SelfAttention(256)
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
model = Autoencoder().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler()  # Enable mixed precision training

def regenerate3(test_image):
    # Load the model for testing
    model_path = "autoencoder_attention_10K_images.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Test the model with a sample image
    # Note: The image is already damaged in app.py before being passed here.
    # So, we no longer need to damage it again here.
    damaged_sample = test_image.unsqueeze(0).to(device)  # No need to damage it again
    predicted_img = model(damaged_sample).cpu().detach().squeeze(0)

    # Compute evaluation metrics
    mse_loss = F.mse_loss(predicted_img, test_image.cpu().squeeze(0)).item()
    psnr_value = 10 * np.log10(1 / mse_loss)
    ssim_value = ssim(
        test_image.cpu().squeeze(0).permute(1, 2, 0).numpy(),
        predicted_img.permute(1, 2, 0).numpy(),
        data_range=1.0,  # Explicitly define the data range for floating point images
        channel_axis=-1  # Explicitly define the channel axis
    )

    l1_loss = F.l1_loss(predicted_img, test_image.cpu().squeeze(0)).item()

    evaluation_metrices_3 = [mse_loss,psnr_value,ssim_value,l1_loss]

    # Convert tensor for display
    restored_image3 = predicted_img
    restored_image3 = restored_image3.squeeze(0).permute(1, 2, 0).numpy()

    # Ensure values are in [0, 1] range
    restored_image3 = np.clip(restored_image3, 0, 1)
    return restored_image3, evaluation_metrices_3