📌 Overview

This repository provides the implementation used to reproduce the denoising, deblurring, and super-resolution results presented in our paper.

📁 Repository Structure
.
├── denoising_test.py        # Denoising script
├── sparse_deblur.py        # Deblurring script
├── superresolution.py      # Super-resolution script
├── model.py                # Model architecture
├── utils.py                # Utility functions
│
├── images/                 # Input images
├── models/                 # Pretrained model weights (.pth)
└── outputs/                # Saved results
⚙️ Installation

All required Python dependencies are listed in requirements.txt.

pip install --upgrade pip
pip install -r requirements.txt
🚀 Usage

Run the scripts from inside the OUR_codes/ directory.

🔹 Denoising
python denoising_test.py
🔹 Deblurring
python pnp_deblurring.py
🔹 Super-Resolution
python pnp_superresolution.py
📦 Outputs

All results are automatically saved in the outputs/ directory.
