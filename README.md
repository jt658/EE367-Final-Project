# Meta-Denoiser: Model-Agnostic Meta-Learning for Image Denoising
------------------------------

## Repository Structure

```
BSDS300/
|   images/             # contains images from the BSDS300 Dataset
|   |   test/           # 200 images used for testing
|   |   train/          # 200 images used for training
|   |   validation/     # 100 images used for validation
BSDSResultsNew/         # meta-denoising results for the BSDS300 Dataset
|   Gaussian_0.1/       # Results for Gaussian noise distribution with variance 0.1
|   Gaussian_0.15/      # Results for Gaussian noise distribution with variance 0.15
|   Gaussian_0.3/       # Results for Gaussian noise distribution with variance 0.3
|   Poisson_15/         # Results for Poisson noise distribution with PEAK parameter 15 (see paper for details)
|   Poisson_5/          # Results for Poisson noise distribution with PEAK parameter 5
|   Poisson_7/          # Results for Poisson noise distribution with PEAK parameter 7
models/
|   basicblock.py       # Provides modules for constructing a DnCNN (ReLU, Conv2d, etc.)
nonlearning_results/    # Results of nonlearning methods on 5 images from BSDS with varying noise distributions
|   100007/             # Results from denoising BSDS300/test/100007.jpg
|                       # Images are named according to noise distribution (G(aussian) or P(oisson)),
|                       # noise parameter (sigma, PEAK), denoising method (BF, NLM, BM3D), and PSNR
|   100099/             # Results from denoising BSDS300/test/10099.jpg
|   103029/             # Results from denoising BSDS300/test/103029.jpg
|   106005/             # Results from denoising BSDS300/test/106005.jpg
|   112056/             # Results from desnoising BSDS300/test/112056
BSDS300Dataset.py       # Implementation of torch.Dataset object for BSDS300 Dataset
Non_Learning_Method...  # Jupyter notebook for running Non-Learning Methods on BSDS dataset
SIDDDataset.py          # Implementation of torch.Dataset object for SIDD Dataset
SIDDbest-model...       # Contains parameters of best denoising model for SIDD Dataset
best-model2000iter...   # Contains parameters of best denoising for BSDS300/500 Dataset
experiment_funcs.py     # Contains PSNR and noise estimation functions, useful for nonlearing methods
metaDenoiser.py         # Contains meta-validation, meta-training, and meta-testing functions for MAML architecture
network_dncnn.py        # Provides implementation of DnCNN
train_and_test.ipynb    # Trains and tests Meta-Denoiser
```
## Instructions
1. Clone this repository: ```$ git clone https://github.com/jt658/EE367-Final-Project.git && cd EE367-Final-Project```
2. Install the necessary dependencies. Download the EE 367 environment from the course [site](http://stanford.edu/class/ee367/HW/HW1.zip). You may also need to ```$ pip install bm3d``` to run Non-Learning-Methods.ipynb.
3. Train and evaluate MetaDenoiser by running the cells in ```train_and_test.ipynb```.
  - See comments in ```train_and_test.ipynb``` for more details
5. Run and evaluate non-learning methods by running the cells in ```Non-Learning-Methods.ipynb```
  - See comments in ```Non-Learning-Methods.ipynb``` for more details

## Authors:
```bibtex
@misc{MetaDenoiser2022,
  author={Tawade, Jessica and Shiv, Rahul},
  title={Few Shot Meta-Learning for Image Denoising},
  month={March},
  year={2022}
```
Contact information:
- Jessica Tawade: jt658@stanford.edu
- Rahul Shiv: rshiv2@stanford.edu

## Citations
We would like to cite the following GitHub repo as a reference for our code:
https://github.com/edwin-pan/MetaHDR
Authors: Edwin Pan and Anthony Vento
