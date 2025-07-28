# Anime Face Generation with GANs

<div align="center">
  <img src="https://img.shields.io/badge/Deep_Learning-GAN-blue?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Task-Image_Generation-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Framework-TensorFlow_Keras-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Dataset-Anime_Faces-red?style=for-the-badge">
</div>

## 📋 Tổng Quan (Overview)

Dự án này triển khai Generative Adversarial Network (GAN) để tạo ra các hình ảnh khuôn mặt anime từ nhiễu ngẫu nhiên. Mô hình được xây dựng với kiến trúc Generator-Discriminator, sử dụng Convolutional layers và các kỹ thuật Deep Learning hiện đại để sinh ra những hình ảnh anime chất lượng cao.

This project implements a Generative Adversarial Network (GAN) to generate anime face images from random noise. The model is built with a Generator-Discriminator architecture, using Convolutional layers and modern Deep Learning techniques to produce high-quality anime images.

## 🎯 Tính Năng Chính (Key Features)

- **Anime Face Generation**: Tạo ra khuôn mặt anime từ vector nhiễu ngẫu nhiên
- **DCGAN Architecture**: Deep Convolutional GAN với Generator và Discriminator
- **Progressive Upsampling**: Tăng dần kích thước từ 8x8 → 64x64 pixels
- **Batch Normalization**: Cải thiện stability và convergence
- **LeakyReLU Activation**: Tối ưu cho Discriminator network

## 🏗️ Kiến Trúc Mô Hình (Model Architecture)

### 1. Generator Network
- **Input**: Latent vector (300 dimensions)
- **Architecture**: Dense → Reshape → Conv2DTranspose layers
- **Output**: RGB image 64x64x3
- **Activation**: ReLU → Tanh (output)

```python
Upsampling Process:
300D noise → 32,768D → 8x8x512 → 16x16x256 → 32x32x128 → 64x64x64 → 64x64x3
```

### 2. Discriminator Network
- **Input**: RGB image 64x64x3
- **Architecture**: Conv2D layers with BatchNorm and LeakyReLU
- **Output**: Binary classification (Real/Fake)
- **Activation**: LeakyReLU → Sigmoid (output)

```python
Downsampling Process:
64x64x3 → 32x32x64 → 16x16x128 → 8x8x128 → 4x4x256 → 2x2x256 → 1 (sigmoid)
```

### 3. Training Configuration
```python
LATENT_DIM = 300
IMAGE_SIZE = 64x64
CHANNELS = 3 (RGB)
BATCH_SIZE = 32
OPTIMIZER = Adam
LOSS = Binary Crossentropy
WEIGHT_INIT = RandomNormal(mean=0.0, stddev=0.02)
```

## 📊 Dataset Information

### Anime Face Dataset
- **Total Images**: 63,565 anime face images
- **Image Size**: Resized to 64x64 pixels
- **Format**: RGB color images
- **Preprocessing**: Normalized to [-1, 1] range
- **Source**: Kaggle Anime Face Dataset

### Data Preprocessing Pipeline
1. **Loading**: Read images from directory
2. **Resizing**: PIL resize to (64, 64)
3. **Normalization**: (pixel - 127.5) / 127.5 → [-1, 1]
4. **Format**: Convert to float32 numpy arrays

## 🛠️ Công Nghệ Sử Dụng (Technology Stack)

### Core Libraries
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **PIL (Pillow)**: Image processing
- **Matplotlib**: Visualization and plotting

### Model Components
- **Conv2DTranspose**: Generator upsampling layers
- **Conv2D**: Discriminator downsampling layers
- **BatchNormalization**: Training stabilization
- **LeakyReLU**: Discriminator activation
- **ReLU**: Generator activation
- **Dense**: Fully connected layers

## ⚙️ Cài Đặt và Sử Dụng (Installation & Usage)

### 1. Dependencies Installation

```bash
pip install tensorflow keras pillow numpy matplotlib
pip install tqdm warnings
```

### 2. Dataset Preparation

```python
# Download Anime Face Dataset from Kaggle
# https://www.kaggle.com/datasets/splcher/animefacedataset

# Directory structure:
# /kaggle/input/animefacedataset/images/
#   ├── image1.jpg
#   ├── image2.jpg
#   └── ...
```

### 3. Data Loading and Preprocessing

```python
import os
import numpy as np
import PIL.Image

# Load and preprocess images
train_images = []
DIR = '/path/to/anime/images'

for path in image_paths:
    img = PIL.Image.open(path)
    img = img.resize((64, 64))
    image = np.array(img)
    train_images.append(image)

# Normalize to [-1, 1]
train_images = np.array(train_images)
train_images = (train_images - 127.5) / 127.5
```

### 4. Generator Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def build_generator():
    model = Sequential(name='generator')
    
    # Dense layer from noise to feature map
    model.add(layers.Dense(8 * 8 * 512, input_dim=LATENT_DIM))
    model.add(layers.ReLU())
    model.add(layers.Reshape((8, 8, 512)))
    
    # Upsampling layers
    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), 
                                     padding='same', kernel_initializer=WEIGHT_INIT))
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), 
                                     padding='same', kernel_initializer=WEIGHT_INIT))
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), 
                                     padding='same', kernel_initializer=WEIGHT_INIT))
    model.add(layers.ReLU())
    
    # Output layer
    model.add(layers.Conv2D(3, (4, 4), padding='same', activation='tanh'))
    
    return model
```

### 5. Discriminator Model

```python
def build_discriminator():
    model = Sequential(name='discriminator')
    
    # Downsampling layers
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', 
                            input_shape=(64, 64, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    
    # Classification layers
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model
```

### 6. Training Process

```python
# Compile models
discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                     loss='binary_crossentropy', metrics=['accuracy'])

# Combined GAN model
discriminator.trainable = False
gan_input = Input(shape=(LATENT_DIM,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = Model(gan_input, gan_output)

gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
           loss='binary_crossentropy')

# Training loop
for epoch in range(epochs):
    # Train Discriminator
    real_images = train_images[random_indices]
    noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
    fake_images = generator.predict(noise)
    
    d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
    
    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
```

## 📈 Kết Quả và Đánh Giá (Results & Evaluation)

### Training Metrics
- **Dataset Size**: 63,565 anime face images
- **Training Time**: ~4-6 hours on GPU (Tesla T4)
- **Batch Size**: 32
- **Image Resolution**: 64x64 pixels
- **Memory Usage**: ~8GB GPU memory

### Model Performance
- **Generator Parameters**: 12,619,203 (48.14 MB)
- **Discriminator Parameters**: ~1.5M parameters
- **Loss Functions**: Binary Crossentropy for both networks
- **Optimizer**: Adam with learning rate 0.0002

### Quality Metrics
- **Visual Quality**: Generated anime faces with distinct features
- **Diversity**: Various hair colors, styles, and facial expressions
- **Resolution**: 64x64 pixel RGB images
- **Realism**: Anime-style artistic representation

### Sample Outputs
```
Generated anime faces show:
- Different hair colors (blonde, brown, black, colorful)
- Various eye styles and colors
- Different facial expressions
- Anime-style artistic features
- Consistent 64x64 resolution
```

## 🔧 Tối Ưu Hóa và Cải Tiến (Optimization & Improvements)

### Current Optimizations
- **Batch Normalization**: Stabilizes training process
- **LeakyReLU**: Prevents dying ReLU problem
- **Weight Initialization**: RandomNormal with proper scaling
- **Dropout**: Prevents discriminator overfitting
- **Adam Optimizer**: Adaptive learning rate

### Future Improvements
- **Progressive Growing**: Gradually increase image resolution
- **Spectral Normalization**: Improve training stability
- **Self-Attention**: Better feature representation
- **Higher Resolution**: 128x128 or 256x256 output
- **Conditional GAN**: Control specific anime attributes
- **StyleGAN Architecture**: State-of-the-art generation quality

## 📁 Cấu Trúc Dự Án (Project Structure)

```
GAN_anime/
├── gan-anime.ipynb                    # Main implementation notebook
├── README.md                          # Project documentation
├── models/
│   ├── generator.h5                  # Trained generator model
│   ├── discriminator.h5              # Trained discriminator model
│   └── gan_complete.h5               # Combined GAN model
├── generated_images/
│   ├── epoch_001.png                 # Generated samples per epoch
│   ├── epoch_050.png
│   └── final_samples.png
├── data/
│   └── anime_faces/                  # Dataset directory
└── utils/
    ├── data_loader.py                # Data loading utilities
    ├── visualization.py              # Plotting functions
    └── model_utils.py                # Model helper functions
```

## 🔍 Chi Tiết Kỹ Thuật (Technical Details)

### Generator Architecture Details
```python
# Layer-by-layer breakdown:
Input: (None, 300)                     # Latent vector
Dense: (None, 32768)                   # 8*8*512 features
Reshape: (None, 8, 8, 512)            # 3D feature map
Conv2DTranspose: (None, 16, 16, 256)  # Upsample to 16x16
Conv2DTranspose: (None, 32, 32, 128)  # Upsample to 32x32
Conv2DTranspose: (None, 64, 64, 64)   # Upsample to 64x64
Conv2D: (None, 64, 64, 3)             # Final RGB output
```

### Discriminator Architecture Details
```python
# Layer-by-layer breakdown:
Input: (None, 64, 64, 3)              # RGB image
Conv2D: (None, 32, 32, 64)            # Downsample
Conv2D: (None, 16, 16, 128)           # Downsample
Conv2D: (None, 8, 8, 128)             # Downsample
Conv2D: (None, 4, 4, 256)             # Downsample
Conv2D: (None, 2, 2, 256)             # Downsample
Flatten: (None, 1024)                 # Flatten features
Dense: (None, 1)                      # Binary output
```

### Loss Functions and Training
```python
# Discriminator Loss
d_loss = -log(D(x)) - log(1 - D(G(z)))

# Generator Loss  
g_loss = -log(D(G(z)))

# Where:
# D(x) = Discriminator output for real images
# G(z) = Generator output from noise z
# D(G(z)) = Discriminator output for fake images
```

## 📚 Tài Liệu Tham Khảo (References)

- [Generative Adversarial Networks (GANs)](https://arxiv.org/abs/1406.2661) - Original GAN Paper
- [Unsupervised Representation Learning with DCGANs](https://arxiv.org/abs/1511.06434) - DCGAN Architecture
- [Improved Training of GANs](https://arxiv.org/abs/1606.03498) - Training Techniques
- [Progressive Growing of GANs](https://arxiv.org/abs/1710.10196) - Progressive Training
- [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) - Kaggle Dataset
- [TensorFlow GAN Tutorial](https://www.tensorflow.org/tutorials/generative/dcgan) - Implementation Guide

## 🤝 Đóng Góp (Contributing)

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

### Contribution Guidelines
- Follow PEP 8 coding standards
- Add docstrings to new functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

Dự án này được phân phối dưới MIT License. Xem file `LICENSE` để biết thêm chi tiết.

## 📞 Liên Hệ (Contact)

- **Author**: kenzn2
- **GitHub**: [@kenzn2](https://github.com/kenzn2)
- **Project**: [GAN_anime](https://github.com/kenzn2/GAN_anime)

## 🙏 Acknowledgments

- Kaggle community for the Anime Face Dataset
- TensorFlow/Keras development team
- Ian Goodfellow and team for the original GAN paper
- DCGAN authors for the convolutional architecture
- Open source community for tools and libraries

## 🎨 Gallery

### Training Progress Examples
```
Epoch 1:    Noise-like images with basic shapes
Epoch 25:   Recognizable facial features emerging
Epoch 50:   Clear anime-style faces with details
Epoch 100:  High-quality anime faces with expressions
```

### Generated Features
- **Hair Styles**: Long, short, wavy, straight, various colors
- **Eye Styles**: Different shapes, colors, and expressions
- **Facial Features**: Varied nose, mouth, and face shapes
- **Artistic Style**: Consistent anime/manga aesthetic

---

<div align="center">
  <b>🎨 Creating Beautiful Anime Faces with the Power of GANs! ✨</b>
</div>