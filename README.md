# Breast Ultrasound Tumor Classification: Heavyweight Transfer Learning vs. Edge-AI 🩺💻

A PyTorch-based Deep Learning pipeline designed to classify breast ultrasound scans into three diagnostic categories: **Normal, Benign, and Malignant**. 

This repository documents an end-to-end medical imaging project that compares two distinct architectural approaches:
1. **The Heavyweight Baseline:** A highly regularized VGG16 model utilizing Transfer Learning (134M Parameters).
2. **The Edge-AI Solution:** A custom, MobileNet-inspired Depthwise Separable CNN built from scratch for low-power medical edge devices (48K Parameters).

---

## 🗂️ Dataset & Preprocessing Pipeline
* **Source:** Breast Ultrasound Images Dataset (BUSI).
* **Size:** 780 patient scans (Normal: 133, Benign: 437, Malignant: 210).
* **Train/Test Split:** 80% Training (624 images) / 20% Unseen Testing (156 images), strictly separated before augmentation to prevent data leakage.
* **Mask Exclusion:** The original dataset mixes raw ultrasounds with radiologist-drawn `_mask.png` segmentation files. A custom `BUSIDataset` PyTorch class was engineered to strictly filter out these masks so the models learn to diagnose based purely on raw ultrasound speckle textures.
* **Transformations:**
  * **Images resized** to `224x224` pixels.
  * **Normalized** using standard ImageNet channel statistics (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`).
  * *(Note: To maintain a strict baseline comparison between models, artificial data augmentation like flips and rotations were excluded).*

---

## 🧠 Model 1: The Heavyweight Baseline (VGG16)
The first approach establishes a strong baseline using Transfer Learning via **VGG16**. Because standard VGG16 easily memorizes a small 780-image dataset, aggressive regularization techniques were required.

### ⚙️ Technical Details:
* **Feature Extraction:** Initialized with ImageNet weights. The foundational blocks were frozen, but **Block 5 was unfrozen** to allow the filters to adapt to medical ultrasound noise.
* **The "Diet" Classifier:** The massive standard fully-connected head (4096 neurons) was replaced with a highly constrained `nn.Linear(in_features=4096, out_features=3)` layer to choke the model's capacity to memorize.
* **Loss Function (Addressing Imbalance):** To prevent the model from blindly guessing the majority "Benign" class, mathematical **Class Weights** were calculated via `sklearn.utils.class_weight.compute_class_weight` and injected directly into the `CrossEntropyLoss` criterion.
* **Optimizer:** Adam (`lr=1e-4`).

### 📊 VGG16 Results:
* **Overall Test Accuracy:** 85%
* **Malignant Class:** 91% Precision | 76% Recall | 83% F1-Score

---

## ⚡ Model 2: The Edge-AI Standard (Depthwise Separable CNN)
To make this AI viable for real-time inference on portable hospital tablets, the VGG16 model was completely replaced with a custom architecture trained from absolute scratch. 

### ⚙️ Architectural Engineering:
* **Depthwise Separable Blocks:** Standard 3x3 convolutions were replaced with a two-phase operation:
  1. **Depthwise Phase:** Uses `groups=in_channels` to force individual 3x3 filters to look *only* at their specific spatial channel.
  2. **Pointwise Phase:** Uses 1x1 filters to mathematically blend the isolated channels into complex features.
* **Progressive Channel Expansion:** * `Prep Block`: 3 -> 32 channels (shrinks to 112x112)
  * `Block 1`: 32 -> 64 channels (shrinks to 56x56)
  * `Block 2`: 64 -> 128 channels (shrinks to 28x28)
  * `Block 3`: 128 -> 256 channels (shrinks to 14x14)
* **Batch Normalization:** `nn.BatchNorm2d` (with `bias=False` in preceding convolutions) was injected after *every single convolutional step* to prevent vanishing gradients during from-scratch training.
* **Global Average Pooling (GAP):** Replaced the standard `Flatten` layer with an `AdaptiveAvgPool2d(1)`. This crushed the 14x14 spatial grid into a 1x1 vector, destroying the small model's ability to cheat via spatial memorization.

### ⏱️ Strict A/B Training Loop:
* **Optimizer:** Standard Adam (`lr=0.001`), identical to the VGG16 baseline.
* **Fair Comparison:** To prove the architectural efficiency, learning rate schedulers and data augmentation were intentionally omitted, forcing the model to rely purely on its depthwise architecture to learn from the data.

### 📊 Edge-AI Results:
Evaluated on the exact same 156-image holdout set using the strict, unassisted baseline, the Depthwise Separable Model achieved an **83.33% Peak Test Accuracy**.

---

## ⚖️ Final Architectural Comparison
How does a custom network built from scratch compare to a 134M-parameter ImageNet behemoth?

| Metric | Heavyweight (VGG16) | Edge-AI (Depthwise) |
| :--- | :--- | :--- |
| **Total Parameters** | 134,268,739 | **48,067** |
| **Parameter Footprint** | Baseline (100%) | **0.035% of VGG16** |
| **Training Method** | Transfer Learning | Trained from Scratch |
| **Peak Test Accuracy** | 84.62% | **83.33%** |

**The Conclusion:** While VGG16 achieved a slightly higher raw accuracy, it requires immense computational overhead. By meticulously separating spatial filtering from channel mixing and implementing GAP + Batch Normalization, the custom Depthwise Separable CNN successfully learned to classify tumors entirely from scratch. We achieved a **99.96% reduction in mathematical footprint** while maintaining a highly reliable **~83% diagnostic accuracy** (trailing the massive ImageNet model by barely 1.3%), proving it is a highly viable, lightweight architecture for immediate deployment on Edge-AI medical devices.

---

## 👨‍💻 Author
**Swarup G L**
* Email: swarupthippyswamy@gmail.com
