import torch
import torch.nn as nn  # ✅ This line ensures nn is properly referenced
from torch.nn.utils import spectral_norm
import torchvision.utils as vutils
import matplotlib.pyplot as plt


class Discriminator(nn.Module):
    def __init__(self, feature_maps=64, channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, feature_maps // 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feature_maps // 2, feature_maps, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False)),

        )

    def forward(self, img):
        return self.model(img).view(-1, 1)


class Generator(nn.Module):
    def __init__(self, latent_dim=100, feature_maps=64, channels=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps, feature_maps // 2, 4, 2, 1, bias=False),  # Extra layer for 128x128
            nn.BatchNorm2d(feature_maps // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps // 2, channels, 4, 2, 1, bias=False),

        )

    def forward(self, z):
        return torch.tanh(self.model(z))



# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define latent dimension (must match training)
latent_dim = 100  # Change based on your training setting

# Load the trained Generator and Discriminator
generator = Generator()  # Ensure this matches your model definition
discriminator = Discriminator()  # Ensure this matches your model definition

generator.load_state_dict(torch.load("generator.pth", map_location=device))
discriminator.load_state_dict(torch.load("discriminator.pth", map_location=device))

generator.to(device).eval()
discriminator.to(device).eval()

# Generate a new image
with torch.no_grad():
    noise = torch.randn(1, latent_dim, 1, 1, device=device)  # Generate a single random noise vector
    generated_image = generator(noise).cpu()

# Evaluate the image with the discriminator
with torch.no_grad():
    disc_output = discriminator(generated_image.to(device))
    confidence = torch.sigmoid(disc_output).item()  # Convert to probability

# Save the generated image
image_path = "generated_sample.png"
vutils.save_image(generated_image, image_path, normalize=True)

# Display the image with discriminator confidence score
plt.imshow(generated_image.squeeze(0).permute(1, 2, 0))
plt.axis("off")
plt.title(f"Discriminator Score: {confidence:.4f}")
plt.show()

print(f"Generated image saved as {image_path}")
