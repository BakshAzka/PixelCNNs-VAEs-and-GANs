# -*- coding: utf-8 -*-
"""
CS437/CS5317/EE414/EE513 Deep Learning Programming Assignment 3
Part 2: Variational Autoencoder (β-VAE) with CelebA Dataset
"""

# <span style="color: #9370DB;">**Submission Guidelines**</span>

- <span style="color: #9370DB;">Please rename the file to `<RollNumber>_PA3_2`.</span>
- <span style="color: #9370DB;">Please also submit a **.py** file of every Notebook. You can do this by exporting your **ipynb** to a Python script.</span>
- <span style="color: #9370DB;">Please submit a zipped folder of both your Jupyter Notebooks and Python script files.</span>

---

<div style="width: 100%; text-align: center;">

<h1 style="color:rgb(244, 244, 245); font-size: 40px; font-weight: bold;">
    <span style="text-decoration-color: white;">THE LAST OF US:
    <span style="color:rgb(130, 89, 214);">DEEP LEARNING EDITION</span></span> 🎮
</h1>

<hr style="height: 10px; width: 100%; background-color: white; border: none; margin-top: 10px; margin-bottom: 10px;">
</div>



<img src="TLOU2.jpg" alt="The Last of Us" style="width: 100%; max-height: 500px; object-fit: cover;">


---

<p style="font-size: 18px;"> In the world of <b>The Last of Us</b>, society has collapsed due to the <b>Cordyceps Brain Infection (CBI)</b>, a fungal outbreak that transforms humans into aggressive, zombie-like creatures known as the <b>Infected</b>. Amidst this chaos, survivors navigate a perilous existence, facing threats from both the Infected and other human factions. </p> <p style="font-size: 18px;"> Our protagonist, <b>Ellie</b>, has endured unimaginable loss and hardship. After tragedy strikes her closest companions, she embarks on a relentless quest for justice. Her journey takes her across the desolate landscapes of a post-apocalyptic America, where she must confront both the horrors of the Infected and the ruthless factions vying for power. </p> <p style="font-size: 18px;"> Ellie's pursuit leads her to the ruins of <b>Hollywood</b>, a city once synonymous with glamour and fame, now a haunting wasteland overrun by the Infected. Intelligence suggests that members of the <b>Washington Liberation Front (WLF)</b> have infiltrated this area, disguising themselves among the hordes to evade detection. To achieve her goal, Ellie must distinguish friend from foe amidst the chaos. </p> <p style="font-size: 18px;"> In this dire scenario, Ellie turns to advanced technology for assistance. She discovers that, much like the Infected emit a distinct scent, humans possess unique facial features that can be analyzed to identify individuals. To leverage this, Ellie employs a <b>Variational Autoencoder (VAE)</b>, a deep learning model capable of learning and disentangling the underlying factors of facial images. By training the VAE on a dataset of celebrity faces, Ellie aims to develop a tool that can differentiate between the Infected, innocent survivors, and hidden WLF operatives. </p> <p style="font-size: 18px;"> This mission is not just about vengeance; it’s about survival, resilience, and the pursuit of justice in a world where morality is no longer black and white. With the VAE as her ally, Ellie embarks on a path fraught with danger, hope, and the unyielding human spirit's fight for redemption. </p>


<div style="width: 100%; text-align: center;">
<h1 style="color:rgb(244, 244, 245); font-size: 40px; font-weight: bold;">
    <span style="text-decoration-color: white;">Let's Start the:
    <span style="color:rgb(130, 89, 214);">Hunt</span></span> 🐾
</h1>
<hr style="height: 10px; width: 100%; background-color: white; border: none; margin-top: 10px; margin-bottom: 10px;">
</div>

# Here is Some Music to help you settle in for the Journey Ahead !
"""

from IPython.display import Audio

# Replace 'your_audio_file.mp3' with your actual file name
audio_file = "03. The Last of Us.mp3"

# Display an audio player in the notebook
Audio(audio_file)

"""---

# <u>**Understanding Variational Autoencoders (VAEs)**</u>

## 📌 What is a Variational Autoencoder?
A **Variational Autoencoder (VAE)** is a **generative model** that learns to encode input data into a **latent space representation** and then reconstruct it back. Unlike traditional **Autoencoders**, VAEs introduce **probabilistic sampling** in the latent space, making them effective for generating new data and disentangling representations.

### 🔹 **Key Idea**
Instead of mapping an input $x$ to a fixed latent vector $z$, VAEs model the latent space as a **distribution**:

- **Encoder**: Learns a **mean** $\mu$ and **variance** $\sigma^2$ for each latent dimension.
- **Latent Space**: Instead of a fixed vector, we **sample** $z$ from a Gaussian distribution:
  
  $$
  z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
  $$

- **Decoder**: Takes the sampled $z$ and reconstructs the input image $\hat{x}$.

This allows for **smooth interpolations** and **better disentangled features** in the latent space.

---

## 📌 What is the **β-Hyperparameter**? 🛠️
The **β-VAE** (Beta Variational Autoencoder) introduces a **hyperparameter** $\beta$ in the loss function to **control the trade-off** between:

- **Reconstruction Quality** (How well the image is reconstructed).
- **Disentanglement** (How independent the latent factors are).

The modified **VAE loss function** becomes:

$$
\mathcal{L} = \mathbb{E}_{q(z|x)} [\log p(x|z)] - \beta D_{KL}(q(z|x) || p(z))
$$

Where:
- $\mathbb{E}_{q(z|x)} [\log p(x|z)]$ = **Reconstruction Loss** (Measures how well the output matches the input).
- $D_{KL}(q(z|x) || p(z))$ = **KL Divergence** (Forces latent space to follow a standard Gaussian distribution).
- $\beta$ = A scaling factor that controls the balance between reconstruction and disentanglement.

---

## 📌 **Why is KL Divergence Important?**
**Kullback-Leibler (KL) Divergence** measures how much one probability distribution **differs** from another. In VAEs, we use it to **regularize the latent space**, ensuring it follows a normal distribution.

### **🔹 General Formula for KL Divergence**
For any two probability distributions \( P(x) \) and \( Q(x) \), the KL Divergence is defined as:

$$
D_{KL}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}
$$

This equation measures how much **information is lost** when \( Q(x) \) is used to approximate \( P(x) \). In simpler terms, it **penalizes deviations** from the target distribution.

### **🔹 KL Divergence in VAEs**
In **VAEs**, our encoder learns a latent representation \( q(z|x) \), which we want to keep **close to** a standard Gaussian prior \( p(z) \). Since both distributions are assumed to be **Gaussian**, we can derive a closed-form solution for KL Divergence:

$$
D_{KL}(q(z|x) \parallel p(z)) = \frac{1}{2} \sum_{i=1}^{d} \left( 1 + \log(\sigma_i^2) - \mu_i^2 - \sigma_i^2 \right)
$$

### **What do these terms represent?**
- **$\mu_i$**: The mean of the latent variable $z_i$.
- **$\sigma_i^2$**: The variance of the latent variable $z_i$.
- **$d$**: The dimensionality of the latent space.
- **$D_{KL}(q(z|x) || p(z))$**: Measures how much the learned latent space deviates from the assumed standard normal distribution.

### **Why do we need KL regularization?**
✅ It prevents **overfitting**.  
✅ It ensures **smooth latent representations**.  
✅ It allows for **better interpolation and sampling**.  

---

<!-- INSPIRATION PAPER HEADER -->
<div style="text-align: center;">
    <h2 style="color:rgb(248, 248, 244); font-size: 40px; font-weight: bold;">
        📜 <b>Inspiration for This Assignment</b>  
    </h2>
</div>

<!-- INSPIRATION PAPER BLOCK -->
<div style="padding: 25px; text-align: center; font-size: 22px; font-weight: bold; color:rgb(242, 244, 245);
            background: rgba(0, 0, 0, 0.6); border-radius: 15px; width: 70%; margin: auto;
            box-shadow: 0px 0px 30px rgba(213, 223, 16, 0.99);">
    
For this assignment, we will be taking **inspiration** from the following paper:  
📖 **[[Disentangled Representation Learning](https://arxiv.org/abs/2211.11695)]**  

Feel free to **read this for insights** and use it as a guide.  

</div>

<hr style="height: 10px; width: 80%; background: linear-gradient(to right, white, #FFD700, white); border: none; margin-top: 15px; margin-bottom: 15px;">

<!-- CHAPTER 1 HEADER -->
<div style="text-align: center;">
    <h1 style="color: #FF8C00; font-size: 55px; font-weight: bold;">
        🍁 <span style="color: white;">Chapter 1</span> -  
        <span style="color: #FFD700;">Fall</span> 🍂
        <span style="color: white;">[30 Points]</span>
    </h1>
    <hr style="height: 10px; width: 80%; background: linear-gradient(to right, white, #FF8C00, white); border: none; margin-top: 10px; margin-bottom: 10px;">
</div>

<!-- INTRODUCTION -->
<div style="padding: 30px; text-align: justify; font-size: 22px; color: white;
            background: rgba(0, 0, 0, 0.6); border-radius: 15px;
            width: 80%; margin: auto;
            box-shadow: 0px 0px 30px rgba(255, 140, 0, 0.6);">
    
<b>The journey begins, not with a model, but with data.</b>  
This time, **you won’t be given a pre-processed dataset**—  
Instead, **you must take control.**  

In this part of the assignment, your task is to **load the raw CelebA dataset and preprocess it yourself.**  
This is an **essential skill** for anyone in this field:  

✅ **Identify, clean, and prepare data**  
✅ **Decide on preprocessing techniques**  
✅ **Gain full creative freedom** based on task requirements  

Your **dataset, your rules.**  
Your **pipeline, your approach.**  
**Let's begin.**  

</div>

<!-- WHAT IS CELEBA? -->
<div style="padding: 10px; text-align: center;">
    <h2 style="color: #FFD700; font-size: 35px; font-weight: bold;">
        🏷️ <b>What is the CelebA Dataset?</b>
    </h2>
</div>

<div style="padding: 30px; text-align: justify; font-size: 22px; color: white;
            background: rgba(0, 0, 0, 0.6); border-radius: 15px;
            width: 80%; margin: auto;
            box-shadow: 0px 0px 30px rgba(255, 140, 0, 0.6);">

CelebA (**Large-scale CelebFaces Attributes Dataset**) is one of the **most widely used datasets** in **computer vision & deep learning research**.  
It contains **202,599 images of celebrities** with **40 attribute labels** per image.  

💡 **Why is CelebA Important?**  
- It enables **facial attribute prediction** (e.g., age, gender, hair color, eyeglasses, etc.).  
- It allows experimentation with **variational autoencoders (VAEs)** for **feature disentanglement**.  
- It is **diverse**, containing **faces of different expressions, poses, and occlusions**.  

💾 **Dataset Structure:**  
📂 **img_align_celeba/** → The actual images (cropped & aligned)  
📄 **list_attr_celeba.txt** → Binary labels (1 or -1) for each image  
📄 **list_eval_partition.txt** → Specifies train/test/val splits  

**For this task, you will decide which files are needed based on your preprocessing choices.**  

</div>

<!-- TASK OVERVIEW HEADER -->
<div style="text-align: center;">
    <h2 style="color: #FFD700; font-size: 35px; font-weight: bold;">
        🎯 <b>Task Overview: Loading & Preprocessing CelebA</b>
    </h2>
</div>

<!-- TASK DETAILS -->
<div style="padding: 30px; text-align: justify; font-size: 22px; color: white;
            background: rgba(0, 0, 0, 0.6); border-radius: 15px;
            width: 80%; margin: auto;
            box-shadow: 0px 0px 30px rgba(255, 140, 0, 0.6);">

📌 **Download the dataset** from this Link: ([CelebA Official Repository](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)).  
💡 **Recommendation:** Use the **Aligned & Cropped Zip Folder**, but the choice is **yours**.  
📖 **Read the ReadMe File available in the dataset directory** to understand **what’s available**.  

### **🛠️ What You Need to Do:**
🔍 **Step 1:** Identify **which files** are necessary for your task. (Hint: You will need the Attributes, The Partitions and The Images themselves)
📥 **Step 2:** Download the required files from the link.  
📂 **Step 3:** Load the dataset into your notebook **(You are free to use ChatGPT, a custom loader, or any other method—Full Creative Freedom).**  
🖼️ **Step 4:** Display **5 random samples** from the **train** split & **5 random samples** from the **test** split.  
📑 **Step 5:** **Examples of what your visualizations should look like are in the FAQs document.**  

⚠️ **IMPORTANT:** The CelebA dataset is **too large** to be used in its entirety.  
🚀 **We strongly recommend reducing it to a subset of 50,000 images** (80-20 split for Train/Eval).  
- **This will significantly reduce training time.**  
- **With 50,000 images, training will take ~40 minutes (CPU) on all β-values.**  
- **Ignoring this will slow down your progress!**  

</div>

<!-- FINAL GRADING WARNING -->
<div style="padding: 30px; text-align: center; font-size: 22px; font-weight: bold; color: red;
            background: rgba(0, 0, 0, 0.6); border-radius: 15px; width: 80%; margin: auto;
            box-shadow: 0px 0px 30px rgba(255, 0, 0, 0.6);">

⚠️ **Your marks for this task depend entirely on your success in loading and visualizing the dataset.**  
If done incorrectly, you **might not be able to complete the rest of the tasks.**  

</div>
"""

# Code Here
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

# Code Here
attr_df = pd.read_csv(
    'list_attr_celeba.txt',
    sep='\s+',
    skiprows=1,
    header=None,
    names=['image_id'] + [f'att_{i}' for i in range(40)]
)

partition_df = pd.read_csv(
    'list_eval_partition.txt',
    sep='\s+',
    header=None,
    names=['image_id', 'split'],
    dtype={'image_id': str}
)
full_data = pd.merge(attr_df, partition_df, on='image_id')
subset = full_data.sample(n=50000, random_state=42).reset_index(drop=True)
train_df = subset.sample(frac=0.8, random_state=42).reset_index(drop=True)
test_df = subset.drop(train_df.index).reset_index(drop=True)
class CelebADataset(Dataset):
    def __init__(self, dataframe, img_dir='img_align_celeba'):
        self.dataframe = dataframe
        self.img_dir = img_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, img_id)
        img = Image.open(img_path).convert('RGB').resize((64, 64))
        img = np.array(img) / 255.0  # Normalize to [0, 1]
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)

if __name__ == '__main__':
    train_dataset = CelebADataset(train_df)
    test_dataset = CelebADataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

# Code Here
def visualize_samples(train_samples=5, test_samples=5):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # Train samples
    for i, img_id in enumerate(train_df['image_id'].sample(train_samples)):
        axes[0, i].imshow(load_image(img_id))
        axes[0, i].axis('off')
        axes[0, i].set_title('Train')

    # Test samples
    for i, img_id in enumerate(test_df['image_id'].sample(test_samples)):
        axes[1, i].imshow(load_image(img_id))
        axes[1, i].axis('off')
        axes[1, i].set_title('Test')

    plt.tight_layout()
    plt.show()
visualize_samples()

print("\n=== Full Data ===")
print("DataFrame shape:", full_data.shape)
print("Columns:", full_data.columns.tolist())
print("First 5 indices:", full_data.index.tolist()[:5])

print("\n=== Training Data ===")
print("Shape:", train_df.shape)
print("Sample indices:", train_df.index.tolist()[:5])

print("\n=== Test Data ===")
print("Shape:", test_df.shape)
print("Sample indices:", test_df.index.tolist()[:5])

"""<div style="text-align: center;">
    <h1 style="color: rgb(255, 183, 76); font-size: 50px; font-weight: bold;">
        ☀️ <span style="color: white;">Chapter 2</span> -  
        <span style="color: rgb(255, 183, 76);">Summer</span> ☀️
        <span style="color: white;">[30 Points]</span>
    </h1>
    <hr style="height: 8px; width: 80%; background: linear-gradient(to right, white, rgb(255, 183, 76), white); border: none; margin-top: 10px; margin-bottom: 10px;">
</div>

<!-- MISSION HEADER -->
<div style="text-align: center;">
    <h1 style="color: #FFA500; font-size: 55px; font-weight: bold;">
        💻 <span style="color: white;">TIME TO START CODING!</span> 💻
    </h1>
</div>

<!-- CINEMATIC INTRO -->
<div style="padding: 25px; text-align: center; font-size: 24px; font-style: italic; color: white;
            background: rgba(0, 0, 0, 0.6); border-radius: 15px; width: 70%; margin: auto;
            box-shadow: 0px 0px 30px rgba(255, 165, 0, 0.6);">
    
"This is where the real challenge begins..."  
Ellie’s journey led her here. **Your journey begins now.**

</div>

<hr style="height: 8px; width: 80%; background: linear-gradient(to right, white, #FFA500, white); border: none; margin-top: 15px; margin-bottom: 15px;">

<!-- MISSION OBJECTIVE TITLE -->
<div style="text-align: center;">
    <h2 style="color: rgb(255, 183, 76); font-size: 35px; font-weight: bold;">
        🎯 <b>Mission Objective:</b> Train a <b>β-VAE</b> to Disentangle Features and Uncover the WLF Operatives
    </h2>
</div>

<!-- STORY-THEMED CODING CHALLENGE -->
<div style="padding: 30px; text-align: justify; font-size: 22px; color: white;
            background: rgba(0, 0, 0, 0.6); border-radius: 15px;
            width: 80%; margin: auto;
            box-shadow: 0px 0px 30px rgba(255, 165, 0, 0.6);">

The **data is ready**, the **tools are set**, and now it’s time to **dive into the unknown**.  
Just like **Ellie**, you must **navigate through uncertainty**,  
experiment with different **β-values**, and **uncover the hidden representations** within faces.

</div>

<!-- FINAL CALL TO ACTION -->
<div style="padding: 20px; text-align: center; font-size: 22px; font-weight: bold; color: rgb(255, 183, 76);">
    You are on your own from this point onwards.  
    <br> <b>Good luck, survivor.</b> 🏹🔥
</div>

## **🔹 Your Task: Complete the β-VAE Implementation**

### **📌 Key Components to Consider**
1. **Encoder (`encode` function)**
   - Takes an input image .
   - Passes it thorugh **Convolution Layers**
   - Outputs two vectors:
     - **Mean (μ):** The predicted mean of the latent distribution.
     - **Log variance (logσ²):** The predicted variance (log-scale) of the latent distribution.

1. **Reparameterization Trick (`reparameterize` function)**
   - Uses the **mean (μ) and log variance (logσ²)** to create a latent vector **z**.
   - Adds **random noise** to make the model **stochastic**.

2. **Decoder (`decode` function)**
   - Takes a latent vector **z**.
   - Passes it through **transposed convolution layers** to reconstruct the image.

3. **Forward Pass (`forward` function)**
   - Takes an input image.
   - **Encodes → Reparameterizes → Decodes** it.
   - Returns **the reconstructed image, mean, and log variance**.

`Note:`: You do not neccseccarily need to follow the structure above , you are free to make whatever functions you require , change constructors values, Adjust Hyperparameters, GO CRAZY (Clicker Symptoms Starting to show xD).Please also look into how different 'latent_dim' values effect your results and why? Currently a default value of 10 is set

# B-VAE Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaVAE(nn.Module):
    def __init__(self, latent_dim=5, beta=5.0, img_size=64):
        """
        Beta-VAE Model

        Args:
            latent_dim (int): Dimension of the latent space (default = 10)
            beta (float): Weighting factor for KL divergence in loss function
        """
        super().__init__()
        self.beta = beta# Hyperparameter for disentanglement
         # 🚀 Define the encoder, decoder, and other layers here
        self.img_size = img_size
        self.channels = 3
        self.latent_dim=latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*8*8, 2*latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*8*8),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Encode input image to latent space representations (μ, logσ²).
        """
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample latent vector z from μ and logσ².
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode latent vector z back to image space.
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Full forward pass: Encode → Reparameterize → Decode
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

"""# B-VAE Loss"""

def beta_vae_loss(recon_x, x, mu, logvar, beta=5.0):
    """
    Beta-VAE Loss Function

    Args:
        recon_x: Reconstructed image
        x: Original image
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: KL divergence weighting factor

    Returns:
        Total loss
    """

    # 🚀 Compute Reconstruction Loss (Mean Squared Error)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)# 🔴 Implement this
    # 🚀 Compute KL Divergence Loss
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))# 🔴 Implement this
    # 🚀 Compute Total Loss
    total_loss = recon_loss + beta * kld_loss# 🔴 Implement this
    return total_loss

"""# 📌 Training Instructions
Now that you have implemented the Beta-VAE, it's time to train it on the CelebA dataset.
### 🎯 Your Tasks
- ✅ Train the Beta-VAE with at least 4 different β values (1, 5, 10, 50).
- ✅ Train for a minimum of 3 epochs (you can train for more).
- ✅ Log and analyze how different β values affect:
- Pleas use `tqdm` library to track progress as it can take up to `45 mins of training` on a CPU
"""

# Training

# 🚀 1. Define Hyperparameters
# 🚀 2. Initialize Model & Optimizer
# 🚀 3. Define Training Function
# 🚀 4. Define Validation Function
# 🚀 5. Train Model for Each β Value
# 🚀 6. Save the Trained Model
def train_model(beta_values=[5], latent_dims=[10], num_epochs=30):  # Increased epochs
    results = {}

    for beta in beta_values:
        for latent_dim in latent_dims:
            print(f"\n=== Training β={beta} | latent_dim={latent_dim} ===")

            # Initialize a new model for each latent_dim
            model = BetaVAE(latent_dim=latent_dim, beta=beta).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Early stopping and learning rate scheduling
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

            best_loss = float('inf')
            for epoch in range(num_epochs):
                model.train()
                epoch_recon_loss = 0.0
                epoch_kl_loss = 0.0
                epoch_total_loss = 0.0

                for batch in train_loader:
                    x = batch.to(device)
                    optimizer.zero_grad()

                    recon_x, mu, logvar = model(x)
                    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
                    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                    total_loss = recon_loss + beta * kl_loss
                    total_loss.backward()
                    optimizer.step()
                    epoch_recon_loss += recon_loss.item()
                    epoch_kl_loss += kl_loss.item()
                    epoch_total_loss += total_loss.item()
                avg_recon = epoch_recon_loss / len(train_loader)
                avg_kl = epoch_kl_loss / len(train_loader)
                avg_total = epoch_total_loss / len(train_loader)
                print(f"β={beta} | z={latent_dim} | Epoch {epoch+1} | Total Loss: {avg_total:.4f} | KL: {avg_kl:.4f} | Recon: {avg_recon:.4f}")
                if avg_total < best_loss:
                    best_loss = avg_total
                    torch.save(model.state_dict(), f'beta_vae_b{beta}_z{latent_dim}.pth')
                else:
                    scheduler.step(avg_total)

            results[(beta, latent_dim)] = {
                'Total Loss': avg_total,
                'KL Divergence': avg_kl,
                'Reconstruction Loss': avg_recon
            }

    # Print table
    print("\n=== Final Results ===")
    print("| Beta | Latent Dim | Total Loss | KL Divergence | Reconstruction Loss |")
    print("|------|-------------|-------------|----------------|----------------------|")
    for (beta, latent_dim), res in results.items():
        print(f"| {beta:^4} | {latent_dim:^11} | {res['Total Loss']:^11.4f} | {res['KL Divergence']:^14.4f} | {res['Reconstruction Loss']:^20.4f} |")

train_model(
    beta_values=[5],
    latent_dims=[6,10,15,20,25,40]

)

"""## 📊 **Logging Your Results**

Once you have trained your **β-VAE** model with different β values, log your results in a table format like the one below.

|   **Beta Value** |   **Total Loss** |   **KL Divergence** |   **Reconstruction Loss** |
|:--------------:|:-------------:|:----------------:|:----------------------:|
| 1  | 1189.31 | 32.7504 | 1156.56 |
| 5  | 1261.73 | 22.6921 | 1148.27 |
| 10 | 1330.06 | 18.7226 | 1142.84 |
| 50 | 1837.18 | 10.5795 | 1308.21 |

### 📌 **How to Interpret Your Table**
- **Total Loss**: The overall loss for the VAE combining both **reconstruction loss** and **KL divergence**.
- **KL Divergence**: How much the learned latent space deviates from a normal Gaussian distribution.
- **Reconstruction Loss**: Measures how well the model reconstructs the input images.

📢 **Your task is to analyze how the KL Divergence and Reconstruction Loss change as β increases!**

"""

## log Your Own results in the same way as shown above
#done in the next task

"""### 🧩 **Visualizing Disentangled Latent Factors**  

Now that you have trained your **β-VAE**, it's time to **explore the latent space** and visualize how different latent factors influence image generation!  

### 🔍 **What You Need to Do:**  
1️⃣ **Select a Base Latent Vector**:  
   - Start with a vector **initialized to zeros** (or another meaningful initialization).  
   - This represents the "average" or neutral latent space representation.  

2️⃣ **Vary One Latent Dimension at a Time**:  
   - Change the value of **one latent dimension** while keeping all others fixed.  
   - Observe how this affects the generated images.  

3️⃣ **Generate Multiple Samples**:  
   - Choose a **suitable range** for variation (e.g., **between -3 and 3**).  
   - Sample **at least 10 variations** across this range.  

4️⃣ **Plot the Results in a Grid**:  
   - Each **row** should represent a different latent dimension.  
   - Each **column** should represent a different sampled value within the chosen range.  
   - Label the axes appropriately for interpretation.  

### 🎯 **Goal:**  
Your goal is to **interpret how different latent dimensions affect image generation**.  
- Do some dimensions **control specific attributes** (e.g., smiling, hair color, face shape)?  
- Are some dimensions **more interpretable** than others?  
- Does increasing β make disentanglement clearer?  

📝 **Experiment with different variation ranges and observe how your β-VAE captures meaningful factors of variation!**  

🔥 **Time to unlock the secrets of the latent space!** 🚀

"""

def visualize_disentangled_latent_factors(model, latent_dim, num_samples=30, variation_range=(-3, 3)):
    """
    Visualize disentangled latent factors by varying one latent dimension.

    Parameters:
    - model: Trained β-VAE model.
    - latent_dim: Number of latent dimensions in the model.
    - num_samples: Number of samples to generate per latent dimension.
    - variation_range: Range of values to vary for each latent dimension.
    """
    model.eval()

    #--------------------------------------------------------------------------------------------------------
    # CODE HERE
    fig = plt.figure(figsize=(num_samples, latent_dim + 1))
    variations = torch.linspace(variation_range[0], variation_range[1], num_samples)
    for i, val in enumerate(variations):
        ax = fig.add_subplot(latent_dim + 1, num_samples, i + 1)
        ax.axis('off')
        ax.text(0.5, 0.5, f"{val:.1f}", ha='center', va='center', fontsize=8, color='red')
    base_z = torch.zeros(latent_dim).to(next(model.parameters()).device)
    with torch.no_grad():
        for dim in range(latent_dim):
            zs = base_z.repeat(num_samples, 1)
            zs[:, dim] = variations

            generated = model.decode(zs).cpu()

            for i in range(num_samples):
                pos = (dim + 1)*num_samples + i + 1
                ax = fig.add_subplot(latent_dim + 1, num_samples, pos)
                ax.axis('off')
                img = generated[i].permute(1, 2, 0).numpy()
                plt.imshow(np.clip(img, 0, 1))

                if i == 0:
                    ax.text(-0.3, 0.5, f'Dim {dim}', rotation=90, va='center', ha='right',
                            fontsize=9, color='blue', transform=ax.transAxes)

    plt.tight_layout()
    plt.show()
    #--------------------------------------------------------------------------------------------------------



# Parametersl
#latent_dim=10
#num_samples = 10  # Number of variations per latent dimension
#variation_range = (-3,3)  # Adjust this range if needed i.e (-3,3)

# Visualize disentangled latent factors
#visualize_disentangled_latent_factors(model, latent_dim=latent_dim, num_samples=num_samples, variation_range=variation_range)

model = BetaVAE(latent_dim=10)
model.load_state_dict(torch.load("beta_vae_b5_z10.pth", map_location=torch.device("mps" if torch.backends.mps.is_available() else "cpu")))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
visualize_disentangled_latent_factors(model, latent_dim=10, num_samples=10, variation_range=(-3, 3))

# Visualize latent_dim=6
model = BetaVAE(latent_dim=6)
model.load_state_dict(torch.load("beta_vae_b5_z6.pth"))
model.to(device)
visualize_disentangled_latent_factors(model, latent_dim=6,num_samples=10, variation_range=(-3, 3))

# For latent_dim=15
model = BetaVAE(latent_dim=15)
model.load_state_dict(torch.load("beta_vae_b5_z15.pth"))
model.to(device)
visualize_disentangled_latent_factors(model, latent_dim=15,num_samples=10, variation_range=(-3, 3) )

# Visualize for latent_dim=12
model_z12 = BetaVAE(latent_dim=12)
model_z12.load_state_dict(
    torch.load("beta_vae_b5_z12.pth",
               map_location=torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
)
model_z12.to(device)
visualize_disentangled_latent_factors(
    model_z12,
    latent_dim=12,
    num_samples=10,
    variation_range=(-3, 3)
)

model_z12 = BetaVAE(latent_dim=12)
model_z12.load_state_dict(
    torch.load("beta_vae_b5_z12.pth",
               map_location=torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
)
model_z12.to(device)
visualize_disentangled_latent_factors(
    model_z12,
    latent_dim=12,
    num_samples=10,
    variation_range=(-1.5, 1.5)
)

# Visualize for latent_dim=20
model_z20 = BetaVAE(latent_dim=20)
model_z20.load_state_dict(
    torch.load("beta_vae_b5_z20.pth",
               map_location=torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
)
model_z20.to(device)
visualize_disentangled_latent_factors(
    model_z20,
    latent_dim=20,
    num_samples=10,
    variation_range=(-3, 3)
)

# Visualize for latent_dim=25
model_z25 = BetaVAE(latent_dim=25)
model_z25.load_state_dict(
    torch.load("beta_vae_b5_z25.pth",
               map_location=torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
)
model_z25.to(device)
visualize_disentangled_latent_factors(
    model_z25,
    latent_dim=25,
    num_samples=10,
    variation_range=(-5, 5)
)

model_z25 = BetaVAE(latent_dim=25)
model_z25.load_state_dict(
    torch.load("beta_vae_b5_z25.pth",
               map_location=torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
)
model_z25.to(device)
visualize_disentangled_latent_factors(
    model_z25,
    latent_dim=25,
    num_samples=5,
    variation_range=(-5, 5)
)

# Visualize for latent_dim=40
model_z40 = BetaVAE(latent_dim=40)
model_z40.load_state_dict(
    torch.load("beta_vae_b5_z40.pth",
               map_location=torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
)
model_z40.to(device)
visualize_disentangled_latent_factors(
    model_z40,
    latent_dim=40,
    num_samples=10,
    variation_range=(-3, 3)
)

"""<div style="text-align: center;">
    <h1 style="color: rgb(173, 216, 230); font-size: 50px; font-weight: bold;">
        ❄️ <span style="color: white;">Chapter 3</span> -
        <span style="color: rgb(173, 216, 230);">Winter</span> ❄️
        <span style="color: white;">[30 Points]</span>
    </h1>
    <hr style="height: 8px; width: 80%; background: linear-gradient(to right, white, rgb(173, 216, 230), white); border: none; margin-top: 10px; margin-bottom: 10px;">
</div>

<!-- SPORE WARNING HEADER -->
<div style="width: 100%; text-align: center;">
    <h1 style="color: rgb(248, 247, 249); font-size: 50px; font-weight: bold;">
        ☣️ <b>WATCH OUT!</b>  
        <span style="color: rgb(130, 89, 214);">
            🦠 SPORES AHEAD!  
        </span>  
        <span style="color: rgb(255, 255, 255);">
            If You Breathe…....... <b>You’re Done!</b> 💀  
        </span>
    </h1>
    <hr style="height: 8px; width: 100%; background: linear-gradient(to right, white, rgb(130, 89, 214), white); border: none; margin-top: 10px; margin-bottom: 10px;">
</div>

<!-- SPORE-INFESTED IMAGES -->
<div style="text-align: center;">
    <img src="TLOU8.jpg" style="width: 48%; height: 1100px; filter: brightness(80%) contrast(110%) saturate(120%); margin-right: 10px; border: 5px solid rgb(130, 89, 214);">
    <img src="TLOU7.jpg" style="width: 48%; height: 1100px; filter: brightness(80%) contrast(110%) saturate(120%); border: 5px solid rgb(130, 89, 214);">
</div>

<!-- SPORE LORE WITH DRAMATIC BACKGROUND -->
<div style="padding: 30px; text-align: justify; font-size: 22px; color: white;
            background: rgba(0, 0, 0, 0.6); border-radius: 15px;
            width: 80%; margin: auto;
            box-shadow: 0px 0px 30px rgba(130, 89, 214, 0.6);">
    
<b>Ellie moves cautiously through the ruins of Hollywood.</b>  
A dead city, swallowed by time. Silent. Abandoned.  

Until she sees it—  

A <b>thick, swirling mist</b> floating in the dim light.  
Particles drift in the air like specks of **death itself**.  

🦠 **SPORES.** Thick. Dense. **Lethal.**  

She tightens her mask.  
Her breath slows. **She has to move forward.**  

Every inch of the air is <b>infected</b>,  
the last whispers of a world long gone.  

</div>

<!-- SUBHEADING: WHAT ARE SPORES? -->
<div style="padding: 10px; text-align: center;">
    <h2 style="color: white; font-size: 30px; font-weight: bold;">
        🦠 <b>What Are Spores?</b>  
    </h2>
</div>

<!-- SPORE LORE WITH DRAMATIC BACKGROUND -->
<div style="padding: 30px; text-align: justify; font-size: 22px; color: white;
            background: rgba(0, 0, 0, 0.6); border-radius: 15px;
            width: 80%; margin: auto;
            box-shadow: 0px 0px 30px rgba(130, 89, 214, 0.6);">
    
<b>According to <span style="color: rgb(130, 89, 214);">The Last of Us lore</span></b>, Spores are  
<b>airborne fungal particles</b> carrying the Cordyceps infection.  
They <b>linger in enclosed spaces</b>, unseen, yet deadly.  

<br>  

💀 <b>Without protection, exposure to Spores is a death sentence.</b> 💀  

<br>

Ellie, immune yet suffocating in the thick air, moves cautiously.  

<br>

<b>The air tastes wrong. The silence is suffocating.</b>  

<br>

Then—  

<br>

🚨 <b>This is where everything changes.</b> 🚨  

</div>


<!-- SUBHEADING: NO MORE SKELETON CODE -->
<div style="padding: 10px; text-align: center;">
    <h2 style="color: white; font-size: 30px; font-weight: bold;">
        💀 <b>NO MORE SKELETON CODE!</b> 💀
    </h2>
</div>

<!-- CINEMATIC BLOCK -->
<div style="padding: 30px; font-size: 22px; color: white;
            background: rgba(0, 0, 0, 0.6); border-radius: 15px;
            width: 80%; margin: auto;
            box-shadow: 0px 0px 30px rgba(130, 89, 214, 0.6);">
    <b>With limited visibility, no outside help, and nowhere to run…</b>  
    Ellie must rely on her instincts.  
    <b>And so must YOU.</b>  
</div>

<!-- FINAL MESSAGE -->
<div style="padding: 30px; font-size: 22px; color: white;
            background: rgba(0, 0, 0, 0.6); border-radius: 15px;
            width: 80%; margin: auto;
            box-shadow: 0px 0px 30px rgba(130, 89, 214, 0.6);">
    She <b>must</b> find the WLF. She <b>must</b> finish what she started.  
    <b>You have unfinished business too.</b>  
</div>

<div style="width: 100%; text-align: center;">
<h1 style="color:rgb(244, 244, 245); font-size: 40px; font-weight: bold;">
    <span style="text-decoration-color: white;">Let's Continue the:
    <span style="color:rgb(130, 89, 214);">Hunt</span></span> 🐾
</h1>
<hr style="height: 10px; width: 100%; background-color: white; border: none; margin-top: 10px; margin-bottom: 10px;">
</div>

## 🛠️ **Your Task: Analyzing Disentangled Representations Across β Values**  

Ellie has learned to distinguish the **Infected from the WLF** using deep learning, but now she must **fine-tune her approach**. Different **β-values** result in different **levels of disentanglement**—your job is to analyze and visualize them.

---

## **🎯 What You Need to Do:**
1️⃣ **Train a β-VAE model** for at least **4 different β values** (1, 5, 10, 50). --> No need to do this again if you already did it above then just use those models.
2️⃣ **Run the training process** for **at least 3 epochs** (you can train for longer).  --> Same procedure as above , i am just repeating the steps incase you do not want to scroll above.
3️⃣ **Save each trained model** separately. --> Ideally you should have done this before but not my fault if you did not , please train again then. No Way around this.
4️⃣ **Visualize the effect of β** on the learned latent space:  
   - Pick a **suitable variation range** (e.g., -3 to 3).  
   - Generate and plot **disentangled latent factors** for **each β value**.  
5️⃣ **Compare the results**—observe **how β affects disentanglement** and reconstruction quality.

---

## **📊 Visualizing the Disentangled Latent Factors**  

For each trained model, you must **vary each latent dimension** and generate corresponding images. The goal is to see **how different features change** when we tweak the latent variables.

🚀 **Your final visualization should include:**
- **Separate plots for each β value**.
- **Variations across latent dimensions**.
- **A comparison of how different β values affect feature disentanglement**.

---

💀 **Ellie's revenge isn't over yet**—and neither is your task!  
🔥 **Push forward, complete the visualizations, and see the true power of β-VAE!**  

---
"""

# Code
train_model(
    beta_values=[1,5,10,50],
    latent_dims=[12],

)

# Code
latent_dim = 12
betas = [1, 5, 10, 50]
num_samples = 10
variation_range = (-3, 3)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

for beta in betas:
    print(f"\n=== Visualizing β={beta} ===")

    # Load model
    model = BetaVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(
        torch.load(f'beta_vae_b{beta}_z{latent_dim}.pth',
        map_location=device)
    )

    # Generate visualization
    visualize_disentangled_latent_factors(
        model,
        latent_dim=latent_dim,
        num_samples=num_samples,
        variation_range=variation_range
    )

"""<!-- EXPLANATION REQUEST HEADER -->
<div style="width: 100%; text-align: center;">
    <h2 style="color: white; font-size: 30px; font-weight: bold;">
        📝 <b>Please Explain Your Results in Full Detail in the Markdown Below</b> 📝
    </h2>
</div>

<!-- INSTRUCTION BLOCK -->
<div style="padding: 30px; text-align: justify; font-size: 22px; color: white;
            background: rgba(0, 0, 0, 0.6); border-radius: 15px;
            width: 80%; margin: auto;
            box-shadow: 0px 0px 30px rgba(130, 89, 214, 0.6);">
    
<b>Provide a thorough breakdown of your results.</b>  
Explain every aspect, including:  

- ✅ **Architecture Choice** – Why did you choose this specific architecture?
- ✅ **Observations** – What do the numbers and trends indicate?  
- ✅ **Patterns** – Are there any significant trends in your results?  
- ✅ **Comparison** – How do your findings compare to expected outcomes?  
- ✅ **Errors/Anomalies** – Any unexpected results? Why might they have occurred?  
- ✅ **Insights** – What conclusions can you draw from these results?  

Take your time to articulate **every detail clearly and concisely**.  

</div>

1)we began with selecting convolutional b-VAE architecture with encoder and decoder tailored for rgb face images.Our goal is to learn disentangled representation like we wanted to control a specific feature of image ie lightning,face pose,orientation and background etc.Initially we used encoder to extract useful features from the images.The encoder gradually reduces the image size while increasing the number of feature maps helping it learn more abstract features.Then it outputs  mean and log-variance for the latent variables.A reparameterization trick is used to sample from the latent space.the decoder uses transposed convolutions to reconstruct the image from the latent representation, hence increasing the spatial size again.Since our encoder has 886,752 paramters and decoder has 271,971 making it to total of 1.16 million paramters ie size is small enough to train efficiently, houwever broader and expressive enough to capture rich facial features.The beta  term controls how much the model prioritizes disentanglement over reconstruction accuracy.A higher beta forcees our model to compress information more, which can lead to more interpretable latent dimensions.it lets us see dimension controls which visual feature by sweeping a value in the latent space and observing how the image changes.
2 and 3)As latent dim increases, KL divergence takes a rise too if you see the trend in the table. This makes sense—more latent dimensions means the model has more freedom to spread the encoded information.our reconstruction loss also improves  as we increase latent dimensions from 20 to 40 which goes from 247.83 to 238.78.After 25, there's a diminishing return—not much of a significant  improvement in recon loss or KL between 25 and 40.Also when we varied our beta the KL divergence drops drastically from 27 to 2.8 as we went from beta=1 to beta-50 where reconstruction gets better.higher β emphasizes disentanglement via stronger regularization  doesnt help reconstruction.When i selected narrow range e.g., -1.5 to 1.5 it did hide meaningful changes like we saw that its difficult to distingush which feature is changing and showing only the prominent ones which are faded in colours too unlike in (-3,3) where we can see them clearly.Also when we increased num samples for latent dim=12 we got some redundant ones too.When we take variation range of (-5,5), latent dimensions now show blurry or unrealistic faces at the far ends of the range.Some images, especially at the edges ( dim 0, dim 4, dim 6, dim 8), appear distorted, noisy, or saturated, making it hard to interpret the changes clearly.Dim 0 controls face orientation and hairstyle volume,Dim 4 transitions from feminine to masculine features,Dim 6 affects background color.Dim 7/8 vary lighting and contrast.But at the ends, these factors become exaggerated or break down, revealing the limits of the model’s learned representation.We chose model with 12 latent dimensions because i found it visually perfect for interpretation like dim 0 controls hair brightness,dim 1 controls face contour,dim 2 controls  eye brow position,dim 3 controls strong variation in background color but possibly disentangled background lighting,Dim 4  controls exposure or brightness,dim 5 contolsface angle  – more frontal vs slight tilt,dim 8 shows strong control over lip size and smile and dim 9 shows overall facial sharpness.When we went with 10 latent dimensions we see clear interpretable variations,good disentanglement but limited variety — some dimensions have very subtle or overlapping effects.With 12 we found a sweet spot between compactness and expressiveness.Latent dimension of 20 captures more detailed variations eg more variation in hairstyle, shadows etc but we see increased redundancy: many dims look visually similar or have low effect.Many look almost identical across the row  indicating unused or underutilized capacity from 13-19.With 25 we saw what we saw with 20 but dilution in our factors too its only good if you consider it for fine grained analysis not otheriwse.higher beta  pushes the model to focus more on regularizing the latent space liek forcing it match standard gaussian  which reduces KL divergence.At b = 1, many dimensions affect multiple features at once like  skin tone and hairstyle mixeindicating less disentanglement.At beta  = 5 and beta  = 10, we begin to see clearer separation. Each latent dimension starts controlling one distinct factor more independently—like lighting, hair darkness, background color, or face shape.At beta= 50, the disentanglement is at its peak though some reconstruction quality drops slightly p due to stronger KL regularization.When we did with 12 latent dimensions and taking 20 samples than 10 we saw Dim 2,3   rows look very similar across all 20 samples. That means changing those dimensions doesn’t affect the image much  they may be redundant or not encoding useful information.We only spot this clearly because we are using 20 samples,it gives us fine-grained resolution.This gives you insights into how well the b-vae is disentangling factors of variation which is the goal of increasing β.With 10 samples, you avoid too much overlap between adjacent samples.
 3)You can see the full effect of each dimension without crowding the grid and helps in visually interpreting more easily too.At b=1,we observe better reconstruction quality, but the latent space is not well disentangled. wheremultiple latent dimensions seem to control similar features like hairstyle or background, making it hard to interpret. This was expected because low beta emphasizes reconstruction over learning independent features.at beta =10 and 5 our disentanglement improves significantly. Each latent dimension starts to represent a clearer, more distinct factoreg brightness, pose
This matches the theory that a moderate beta  balances both disentanglement and reconstruction.when beta is 50,disentanglement becomes strong, but reconstruction quality suffers ie faces become blurry or lose details. This was expected, as a very high b over-penalizes the KL term, forcing the latent space to be highly compressed and clean, but at the cost of realism and noise (that explains the very low KL (2.88) and huge reconstruction loss).Dimensions start getting repetitive ie in 4 and 10. Some dims barely change anything, which indicates redundancy. That’s why KL drops more (11.55), and visuals are still okayish but slightly less interpretable than b=5.our visuals reflect expected trade-offs of β-VAE.beta= 5 provides the best disentanglement and interpretability.b= 1 and b = 50 show the weaknesses of under/over-regularization.I would say beta 5 to be the middle ground which captures  each dimension controls eg skin tone, hair color, brightness etc.
4)the unexpected observation was redundancy in latent dimensions, especially when using higher values like latent dim=20 or latent dim=40, even though our dataset is rich and diverse enough to support that complexity which might be due to insufficient regularixation. The encoder may also have distributed similar features across multiple dimensions.the dataset might not require that many dimensions to encode meaningful variations and when setting num_samples=20, adjacent samples start to look nearly identical, especially near the center of the latent range.The variation_range [-2.5, 2.5] is not wide enough to generate clearly distinct representations at such fine granularity or it could be  extreme ends show change, and the middle is mostly flat.At certain latent values (especially in Dim 11), we noticed abrupt brightness shifts where the background or face turns either completely white or black, even though other dimensions behave smoothly.Some latent dimensions unexpectedly affected background color (turning blue or orange), even though background should ideally stay fixed.This likely means the model has picked up dataset-specific artifacts like common background colors in the CelebA dataset and embedded them in the latent space.at b=50 several latent dimensions became almost inactive or redundant, showing minimal change during traversal ie, Dim 0–4 or produced very subtle, barely noticeable variations.Instead of more disentangled, the model became under-expressive, prioritizing KL loss so heavily that it sacrificed reconstruction quality and usable latent capacity.
5)As b increases, the KL divergence term dominates, encouraging disentangled representations  but at the cost of higher reconstruction loss.Visual traversals show clearly interpretable latent factors — each row modifies one aspect such as hair texture, skin tone, head pose, background, or lighting, with minimal redundancy. When increasing latent dimensions beyond 20, we saw visual redundancy  several latent factors showed little or no variation.Unexpected disentanglement at b=1, and background-related factors being captured, suggest that the structured and consistent nature of CelebA  makes it easier for the model to learn clean factors even when the regularization is low.Despite using a relatively compact encoder-decoder architecture with just 4 conv layers and 3 transposed conv layers, our  model successfully captured and disentangled meaningful features like skin colour,haircolors and facial brightness etc.
our results reinforce the original b-VAE paper’s claims:
Higher beta  leads to better disentanglement but worse reconstruction.Increasing  b greater than1 forces the model to prioritize learning disentangled latent factors, at the cost of reconstruction quality.Reconstruction loss increased significantly  from 254 to 522, meaning the decoder lost some ability to reconstruct fine details.This confirms the expected trade-off described in the paper — higher disentanglement comes with poorer reconstruction.With the right beta and architecture, each latent variable should control a single interpretable factor of variation which matches with ours too.Extremely high beta values can overly constrain the latent space, leading to loss of useful information.At b= 50, latent units barely changed outputs.Reconstructions were blurry and nearly identical, with minimal variation across traversals.Total loss shot up due to high reconstruction loss ie 522.This supports the paper’s conclusion that excessive regularization can harm it.There exists an optimal b that provides good disentanglement with tolerable reconstruction error where our best  trade-off came at b = 5.

<!-- CHAPTER 3 HEADER -->
<div style="text-align: center;">
    <h1 style="color: rgb(255, 165, 0); font-size: 50px; font-weight: bold;">
        🍂 <span style="color: white;">Chapter 4</span> -  
        <span style="color: rgb(255, 165, 0);">Autumn</span> 🍂
        <span style="color: white;">[10 Points]</span>
    </h1>
    <hr style="height: 8px; width: 80%; background: linear-gradient(to right, white, rgb(255, 165, 0), white); border: none; margin-top: 10px; margin-bottom: 10px;">
</div>

<!-- REFLECTION QUESTIONS BLOCK -->
<div style="padding: 30px; text-align: justify; font-size: 22px; color: white;
            background: rgba(0, 0, 0, 0.6); border-radius: 15px;
            width: 80%; margin: auto;
            box-shadow: 0px 0px 30px rgba(255, 165, 0, 0.6);">
    
<b>Now that you have reached Autumn, it's time to reflect.</b>  
Answer the following questions thoughtfully:

## 🔍 Reflection Question 1
### How does increasing β affect disentanglement and reconstruction quality? Please provide a detailed response (Max 100 Words)

Ans:In my results, increasing b from 1 to 50 clearly improved disentanglement but worsened reconstruction quality. At b = 1, the reconstructions were sharp but showed poor disentanglement. As b increased to 5 and 10, clearer changes in individual latent dimensions emerged, showing better disentanglement. However, at b = 50, reconstructions became blurry and repetitive, as reflected in the high reconstruction loss (522.38). This shows that higher b forces the model to prioritize disentanglement (KL dec from 27.2 to 2.88), but at the cost of useful detail in generated images.

## 🔍 Reflection Question 2
### Identify and discuss the limitations of β-VAE, such as loss of fine-grained details in reconstructions or failure to disentangle specific factors. Please provide a detailed response (Max 100 Words)

Ans:

A key limitation of B-VAE is the trade-off between disentanglement and reconstruction quality. In my Case increasing b to 50 resulted in very high reconstruction loss (522.38), causing loss of finegrained details like background and facial features. Although disentanglement improved, some latent dimensions showed redundancy or entangled factors like similar variations across dims. Additionally, at lower b like at 1, reconstructions were sharp but failed to isolate independent factors. This highlights that while b-VAE promotes interpretability, too high a b can harm realism and fail to disentangle all factors present in csuch complex datasets like CelebA.

<!-- FINAL MISSION HEADER -->
<div style="text-align: center;">
    <h1 style="color: rgb(255, 215, 0); font-size: 55px; font-weight: bold;">
        🎯 <span style="color: white;">Mission Accomplished!</span>  
    </h1>
    <h2 style="color: rgb(255, 140, 0); font-size: 40px; font-weight: bold;">
        <b>The Last of Us: DL Edition Ends Here</b> 🎮🔥
    </h2>
    <hr style="height: 10px; width: 80%; background: linear-gradient(to right, white, rgb(255, 140, 0), white); border: none; margin-top: 10px; margin-bottom: 10px;">
</div>

<!-- STORY CLOSURE -->
<div style="padding: 30px; text-align: justify; font-size: 22px; color: white; background: rgba(0, 0, 0, 0.7); border-radius: 15px; width: 80%; margin: auto; box-shadow: 0px 0px 40px rgba(255, 140, 0, 0.7);">
<b>A Tale of Vengeance, Survival, and AI</b>

The journey was <b>brutal</b>, the road was <b>long</b>, but <b>Ellie never faltered</b>.
Through <b>blood, sweat, and deep learning</b>, she <b>saw her mission through to the end</b>.

With the power of Variational Autoencoders (VAEs), she uncovered hidden enemies, infiltrating their ranks and exposing those who lurked in the shadows.
The very faction that had <b>taken everything from her</b>—that shattered the last remnants of her world—had now been <b>brought to justice</b>.

💔 <b>For those she lost. For those still fighting.</b> 💔

</div> <!-- SUBHEADING: HOW DID ELLIE USE DL? --> <div style="padding: 10px; text-align: center;"> <h2 style="color: white; font-size: 30px; font-weight: bold;"> 🔬 <b>How Did Ellie Use Deep Learning to Complete Her Mission?</b> </h2> </div> <!-- CINEMATIC STORY BLOCK --> <div style="padding: 30px; text-align: justify; font-size: 22px; color: white; background: rgba(0, 0, 0, 0.7); border-radius: 15px; width: 80%; margin: auto; box-shadow: 0px 0px 40px rgba(255, 140, 0, 0.7);">
🔥 <b>In a world where survival depends on identifying the enemy</b>, Ellie turned to deep learning for answers.

- ✅ Using a β-VAE, she <b>disentangled hidden facial attributes</b> and <b>revealed those who wished to stay hidden</b>.
- ✅ She experimented with different <b>β values</b>, learning how the model traded off between <b>accuracy and interpretability</b>.
- ✅ With enough training, she <b>decoded the WLF's disguises</b>, proving that <b>no mask, no shadow, no camouflage could stop the power of AI</b>.

Her mission was not just about revenge, but about ensuring no one else suffered the same fate.
With the memory of the fallen in her heart and science in her hands, she rewrote fate itself.

⚔️ <b>Justice has been served.</b>
🎮 The Last of Us: AI Edition has come to a close.

🥀 For those we've lost. For those still standing. 🥀

</div>


<!-- MARKING CRITERIA -->
<div style="padding: 10px; text-align: center;">
    <h2 style="color: white; font-size: 35px; font-weight: bold;">
        📜 <b>Marking Criteria & Submission Guidelines</b>  
    </h2>
</div>

<!-- CRITERIA BLOCK -->
<div style="padding: 30px; text-align: justify; font-size: 22px; color: white;
            background: rgba(0, 0, 0, 0.7); border-radius: 15px;
            width: 80%; margin: auto;
            box-shadow: 0px 0px 40px rgba(255, 140, 0, 0.7);">

## 📊 **How Will You Be Evaluated?**  

Unlike previous parts of the assignment, there are **no strict benchmarks** (e.g., specific loss values).  
This part is **open-ended**, meaning **multiple correct implementations exist**.  

✅ **We will assess your work based on the following:**  
- **Architecture choices**: How well have you structured your VAE model?  
- **Results**: How effectively does your model learn meaningful latent representations?  
- **Quality of Visualizations**: The true measure of your model's success is in its ability to **clearly separate different features**. Your plots should **demonstrate disentanglement effectively**.  

</div>

<!-- SUBMISSION REQUIREMENTS -->
<div style="padding: 10px; text-align: center;">
    <h2 style="color: white; font-size: 30px; font-weight: bold;">
        📂 <b>Submission Requirements</b>  
    </h2>
</div>

<!-- SUBMISSION INSTRUCTIONS BLOCK -->
<div style="padding: 30px; text-align: justify; font-size: 22px; color: white;
            background: rgba(0, 0, 0, 0.7); border-radius: 15px;
            width: 80%; margin: auto;
            box-shadow: 0px 0px 40px rgba(255, 140, 0, 0.7);">

### 1️⃣ **Model Checkpoints**  
📌 **Save all 4 trained models** (one for each β value) and upload them to **Dropbox** under your name.  
- **File format**: `.zip`  
- **Naming format**: `<RollNumber>_PA3_2_Models.zip`  

### 2️⃣ **Notebook Submission**  
📌 Submit **this notebook and its Python file** (`.ipynb` and `.py`) on **LMS** with the rest of the assignment.  

</div>

<!-- FINAL CONGRATULATIONS BLOCK -->
<div style="padding: 30px; font-size: 22px; color: white;
            background: rgba(0, 0, 0, 0.7); border-radius: 15px;
            width: 80%; margin: auto;
            box-shadow: 0px 0px 40px rgba(255, 140, 0, 0.7); text-align: center; font-weight: bold;">
    
🔥 **If you’ve made it this far, congratulations!**  
You’ve successfully implemented a **β-VAE**, explored its impact, and helped Ellie on her quest.  

👑 <b>You are now a true deep learning survivor.</b>  

🚀 **See you in the next adventure!**  

</div>

~ A Saad Haroon Production
"""