
## 📋 Overview

This repository contains implementations of three core generative models in deep learning as part of the CS437/CS5317/EE414/EE513 Deep Learning course. The assignment explores different approaches to image generation and representation learning using popular deep learning architectures.

## 🎯 Project Components

### 1. PixelCNN Implementation (Part 1)
**File:** `PA3_1.ipynb` / `PA3_1.py`

An implementation of PixelCNN for autoregressive image generation using the MNIST dataset.

**Key Features:**
- Custom `PixelConv` layers with masked convolutions (Type A and Type B)
- Residual blocks for improved training stability
- Autoregressive pixel-by-pixel image generation
- Binary image preprocessing and normalization
- Model trained on 28×28 handwritten digit images

**Architecture Highlights:**
- Masked convolutions ensure causal pixel dependencies
- Multiple residual blocks for deep feature extraction
- Parameter count kept under 600,000 for efficiency
- Sequential pixel generation from top-left to bottom-right

### 2. Variational Autoencoder (β-VAE) (Part 2)
**File:** `PA3_2.ipynb` / `PA3_2.py`

A creative implementation of β-VAE on the CelebA dataset, themed around "The Last of Us" game universe.

**Key Features:**
- Implementation of Variational Autoencoders with β-hyperparameter tuning
- Disentangled representation learning for facial attributes
- Custom dataset preprocessing pipeline for CelebA
- Exploration of different β values for reconstruction vs. disentanglement trade-off
- KL divergence regularization for smooth latent space

**Dataset:**
- CelebA (Large-scale CelebFaces Attributes Dataset)
- 50,000 image subset (80-20 train/test split)
- Images resized to 64×64 pixels
- 40 facial attribute labels available

**Technical Concepts:**
- Latent space sampling with reparameterization trick
- β-weighted KL divergence for controlled disentanglement
- Facial feature interpolation and manipulation

### 3. Generative Adversarial Network (GAN) (Part 3)
**File:** `PA3_3_.ipynb` / `pa3_3_.py`

A GAN implementation for generating MNIST handwritten digits with latent space exploration.

**Key Features:**
- Generator network: transforms random noise into realistic digit images
- Discriminator network: distinguishes between real and fake images
- Adversarial training process with balanced loss functions
- Latent space interpolation between different digits
- Vector arithmetic operations in latent space

**Architecture:**
- **Generator:** 4-layer fully connected network with BatchNorm and LeakyReLU
- **Discriminator:** 4-layer fully connected network with Dropout and LeakyReLU
- Latent dimension: 64
- Binary Cross-Entropy loss for both networks

**Capabilities:**
- Generate new handwritten digits from random noise
- Smooth interpolation between two digit images
- Latent space vector arithmetic (e.g., generating variations)

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** (for PixelCNN)
- **PyTorch** (for GAN and VAE)
- **NumPy** - Numerical computations
- **Matplotlib/Seaborn** - Data visualization
- **Pandas** - Data manipulation
- **PIL/Pillow** - Image processing

## 📊 Datasets

### MNIST
- 60,000 training images + 10,000 test images
- 28×28 grayscale images of handwritten digits (0-9)
- Preprocessed with binarization (threshold = 0.5)

### CelebA
- 50,000 images (subset from 202,599 total images)
- 64×64 RGB celebrity face images
- 40 binary attribute labels per image
- 80-20 train/test split

## 🚀 How to Run

### Prerequisites
```bash
# Install required packages
pip install tensorflow torch torchvision numpy matplotlib pandas seaborn pillow tqdm
```

### Running the Projects

#### 1. PixelCNN (Part 1)
```bash
# Run the Jupyter notebook
jupyter notebook PA3_1.ipynb

# Or run the Python script
python PA3_1.py
```

#### 2. β-VAE (Part 2)
```bash
# Download CelebA dataset first from:
# https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

# Place the following files in the project directory:
# - img_align_celeba/ (folder with images)
# - list_attr_celeba.txt
# - list_eval_partition.txt

# Run the notebook
jupyter notebook PA3_2.ipynb

# Or run the Python script
python PA3_2.py
```

#### 3. GAN (Part 3)
```bash
# Run the Jupyter notebook
jupyter notebook PA3_3_.ipynb

# Or run the Python script
python pa3_3_.py
```

## 📈 Results

### PixelCNN
- Successfully generates binary MNIST digits pixel-by-pixel
- Maintains autoregressive dependencies throughout generation
- Model converges with stable training

### β-VAE
- Achieves disentangled representations of facial features
- Different β values show trade-off between reconstruction quality and disentanglement
- Smooth interpolation between different faces in latent space

### GAN
- Generates realistic handwritten digits after ~50 epochs
- Discriminator and Generator losses stabilize during training
- Smooth transitions achieved through latent space interpolation
- Saved models: `generator.pth` and `discriminator.pth`

## 📁 Project Structure

```
PixelCNNs-VAEs-and-GANs/
│
├── PA3_1.ipynb                   # PixelCNN notebook
├── PA3_1.py                      # PixelCNN Python script
│
├── PA3_2.ipynb                   # β-VAE notebook
├── PA3_2.py                      # β-VAE Python script
│
├── PA3_3_.ipynb                  # GAN notebook
├── pa3_3_.py                     # GAN Python script
│
├── generator.pth                 # Trained GAN generator weights
├── discriminator.pth             # Trained GAN discriminator weights
│
├── loss_plot.png                 # GAN training loss visualization
│
└── README.md                     # This file
```

## 🎓 Learning Outcomes

Through this assignment, I gained hands-on experience with:

1. **Autoregressive Models**: Understanding how PixelCNN generates images sequentially
2. **Latent Space Learning**: Exploring VAE's ability to learn meaningful representations
3. **Adversarial Training**: Implementing the minimax game between Generator and Discriminator
4. **Model Architecture Design**: Building custom layers and residual blocks
5. **Dataset Preprocessing**: Handling real-world datasets (CelebA) with custom pipelines
6. **Hyperparameter Tuning**: Exploring β-hyperparameter effects on disentanglement
7. **Loss Function Design**: Implementing reconstruction loss, KL divergence, and adversarial losses

## 🔍 Key Concepts Demonstrated

- **Masked Convolutions**: Ensuring causal dependencies in autoregressive generation
- **Reparameterization Trick**: Enabling backpropagation through stochastic sampling
- **KL Divergence**: Regularizing latent space to follow standard normal distribution
- **Adversarial Training**: Balancing Generator and Discriminator learning
- **Latent Space Interpolation**: Exploring continuous representations
- **Disentangled Representations**: Separating independent factors of variation



