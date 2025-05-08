# AI-Powered Human Face Reconstruction

An advanced deep learning project that restores damaged or occluded human facial images using convolutional autoencoders and attention-based architectures. The system supports web-based interaction via a Flask dashboard and a Telegram bot for real-time restoration.

## 🚀 Project Overview

This project aims to reconstruct facial images suffering from occlusions or distortions using deep learning techniques. Three different autoencoder-based models were developed and evaluated using metrics like PSNR, SSIM, MSE, and L1 Loss. A user-friendly Flask interface and Telegram bot were integrated to provide an accessible and interactive experience.

## 🧠 Models Implemented

- **Model 1:** Baseline Convolutional Autoencoder
- **Model 2:** Enhanced Autoencoder with Output Normalization
- **Model 3:** Attention-Augmented Autoencoder

## 🗂️ Dataset

- **Source:** CelebA dataset
- **Preprocessing Includes:**
  - Facial damage simulation
  - Augmentation & normalization
  - 80:20 train-test splitting
  - Image resizing (128x128)

## 🛠️ Tech Stack

- **Languages & Libraries:** Python, NumPy, OpenCV, Matplotlib, PyTorch
- **Frameworks:** Flask (for web interface), Telebot (Telegram bot)
- **Hardware:** GPU-enabled (NVIDIA recommended for model training)

## 🖥️ System Modules

- **Preprocessing Module**
- **Model Training & Evaluation**
- **Web Dashboard (Flask)**
- **Telegram Bot for Face Restoration**

## 💻 Installation & Setup

1. **Clone the Repository**
   git clone https://github.com/umangmtiwari/AI-Powered-Human-Face-Reconstruction.git
   cd face-reconstruction

2. **Create and Activate Virtual Environment**

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies**

   pip install -r requirements.txt

4. **Download the Dataset**

   * Download the CelebA dataset from [official source](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
   * Place it in the `/data` folder

5. **Run Flask App**

   python app.py

6. **Run Telegram Bot**

   python mobile.py

## 📷 Screenshots

![image](https://github.com/user-attachments/assets/2b753e30-b52b-4585-b2fd-410153bf6a0e)

![image](https://github.com/user-attachments/assets/48d9730c-4c5a-44c4-9ec2-a4fe1e3abf74)

![image](https://github.com/user-attachments/assets/265772ab-094a-420d-a9ab-2210f38afc06)

![image](https://github.com/user-attachments/assets/56931bed-56b3-4d8b-b8a4-930f2ce2d170)

![image](https://github.com/user-attachments/assets/d188254c-7057-4b9e-9312-ddf260e9ab5d)

![image](https://github.com/user-attachments/assets/05c2731f-712f-4bca-a59c-c3786151deda)


## 🤖 Telegram Bot Features

* Upload a damaged face directly via Telegram
* Receive reconstructed versions from all three models
* Compare and download results

## 🧪 Testing

* Manual visual inspection
* Quantitative metrics: PSNR, SSIM, MSE, L1
* Batch testing and output comparison

## 📜 License

This project is licensed under the MIT License.

## 🤝 Contributions

Contributions are welcome! Please open issues or submit pull requests for improvements or additional features.
