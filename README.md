This repository contains the code used to reproduce the denoising, deblurring, and super-resolution results in our paper.


#Folder Structure
│
├── denoising_test.py
├── sparse_deblur.py
├── superresolution.py
├── model.py
├── utils.py
│
├── images/      # input images for each 
├── models/      # pretrained weights (.pth) here
└── outputs/     # results saved here

#All required Python packages are listed in requirements.txt. Install them with
pip install --upgrade pip
pip install -r requirements.txt

##Running the Code

Run each script from inside OUR_codes/:

#Denoising
python denoising_test.py

#Deblurring
python pnp_deblurring.py

#Super-Resolution
python pnp_superresolution.py


#Outputs will be saved automatically inside:
outputs/
