# Breast Ultrasound Classification Using VGG16 🩺💻

[cite_start]A Deep Learning pipeline built in PyTorch to classify breast ultrasound scans into three diagnostic categories: **Normal, Benign, and Malignant**[cite: 6]. This project tackles severe class imbalance and overfitting on a limited medical dataset using custom regularization and weighted loss functions.

## 📋 Project Overview
[cite_start]This project leverages Transfer Learning via the **VGG16** Convolutional Neural Network[cite: 11]. While the dataset is small (780 images) and heavily skewed toward "Benign" cases, this pipeline successfully prevents majority-class bias and achieves high recall on critical "Malignant" classifications.

**Key Achievements:**
* Handled severe dataset class imbalance without artificially synthesizing blurry medical images.
* Prevented massive model memorization (overfitting) using extreme regularization.
* Achieved an 85% Overall Test Accuracy with an 83% F1-Score specifically for Malignant tumor detection.

## 🗂️ Dataset Details
* **Source:** Breast Ultrasound Images Dataset (BUSI)[cite: 7].
* **Size:** 780 patient scans[cite: 7].
* **Classes:** Normal (133), Benign (437), Malignant (210)[cite: 6].
* **Data Processing:** The original BUSI dataset mixes raw ultrasound images with radiologist-drawn `_mask.png` segmentation files[cite: 8]. [cite_start]A custom PyTorch `DataLoader` was built to strictly filter out the masks, ensuring the model diagnoses based purely on raw ultrasound textures without data leakage[cite: 8, 9].

## 🧠 Architecture & Engineering Strategy

### 1. The Foundation (Transfer Learning)
* Initialized VGG16 with standard ImageNet weights to utilize its pre-learned edge and contrast detection capabilities.
* Froze the early foundational layers, but **unfroze Block 5** to allow the model to fine-tune its "eyes" specifically to medical ultrasound speckle noise.

### 2. Defeating Overfitting (The "Diet" Classifier)
VGG16's standard 138-million parameter architecture easily memorizes a 780-image dataset. To force true learning, the standard classifier was replaced with a custom, highly regularized block:
* Squeezed the dense layers down to just 256 neurons.
* Applied **50% Dropout** to randomly disable neurons during training.
* Implemented heavy **Data Augmentation** (Rotations, Flips, ColorJitter) so the model never sees the exact same pixel grid twice.
* Added **L2 Weight Decay** via the Adam optimizer to penalize over-confidence.

### 3. Solving Class Imbalance
To stop the model from taking the "easy route" and blindly guessing the majority class ("Benign"), **Class Weights** were calculated based on the inverse frequency of the training set. These weights were fed directly into the `CrossEntropyLoss` function, heavily mathematically penalizing the model for missing the rare "Malignant" and "Normal" scans.

## 📊 Results & Evaluation
Overall Accuracy on an imbalanced dataset is deceptive; the true measure of a medical model is its Precision and Recall on minority classes. Evaluated on a locked 20% unseen test split (156 images):

* **Overall Test Accuracy:** 85%
* **Malignant Class:** 91% Precision | 76% Recall | 83% F1-Score
* **Normal Class:** 71% Precision | 93% Recall | 81% F1-Score
* **Benign Class:** 87% Precision | 88% Recall | 87% F1-Score

The 91% precision rate on Malignant tumors indicates that when the model flags a tumor as dangerous, it is highly reliable. The 93% recall on Normal scans proves the Class Weights successfully forced the model to identify the rarest dataset class.

## 🛠️ Tech Stack & Libraries
* **Framework:** PyTorch (`torch`, `torch.nn`, `torch.optim`)
* **Computer Vision:** `torchvision` (Transforms, Models)
* **Data Handling:** `torch.utils.data` (Dataset, DataLoader), `PIL`, `NumPy`
* **Visualization:** `Matplotlib`, `Seaborn`, `scikit-learn` (Confusion Matrix & Classification Report)
* **Environment:** Kaggle Notebooks / Jupyter

## 🚀 How to Run
1. Clone this repository.
2. Download the BUSI dataset from Kaggle and place the unzipped folder in your working directory.
3. Update the `DATA_DIR` path variable in the notebook to point to your local dataset folder.
4. Run the Jupyter Notebook cells sequentially. 
*(Note: A CUDA-enabled GPU is highly recommended for training).*

## 👨‍💻 Author
**Swarup G L**
* Student, IIIT Raichur
* GitHub: [@Swarupgl](https://github.com/Swarupgl)
* Email: swarupthippyswamy@gmail.com
