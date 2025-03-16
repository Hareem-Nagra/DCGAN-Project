# DCGAN-Project
# DCGAN for Animal Faces HQ (AFHQ) Dataset

## Project Overview
This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) in PyTorch using the Animal Faces HQ (AFHQ) dataset. The goal is to generate realistic images of animal faces by training a GAN with convolutional layers and batch normalization.

## Dataset
The AFHQ dataset consists of high-quality images of animal faces categorized into three classes:
- Cats
- Dogs
- Wild animals

The dataset can be downloaded from [AFHQ Dataset](https://github.com/clovaai/stargan-v2). It is preprocessed and loaded into PyTorch using the `torchvision.datasets.ImageFolder` class.

## Model Architecture
DCGAN consists of two main components:
1. **Generator**: A deep convolutional neural network that generates realistic images from random noise.
2. **Discriminator**: A convolutional neural network that classifies images as real or fake.

Key features of the architecture:
- Uses batch normalization and LeakyReLU in the discriminator.
- Uses batch normalization and ReLU in the generator.
- Employs transposed convolution layers for upsampling in the generator.
- Uses the Adam optimizer for training.

## Training
- The model is trained using the **Binary Cross-Entropy Loss (BCELoss)**.
- **Adam optimizer** is used with learning rate 0.0002 and beta values (0.5, 0.999).
- The training loop alternates updates between the generator and discriminator.
- Images are saved at intervals to visualize training progress.

## Dependencies
- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- NumPy
- PIL
- tqdm

## Running the Code
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/dcgan-afhq.git
   cd dcgan-afhq
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```sh
   python train.py
   ```
4. Generated images will be saved in the `results/` directory.

## Results
During training, generated images improve progressively. The discriminator learns to differentiate between real and fake images, while the generator refines its outputs to produce high-quality animal faces.



