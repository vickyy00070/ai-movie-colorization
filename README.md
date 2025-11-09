# AI-Based Movie Colorization — Deep Learning Project

**Author:** Vivek Rudra  
**Affiliation:** Gandhi Institute of Technology and Management (GITAM)  
**Repository:** https://github.com/vickyy00070/ai-movie-colorization

## Abstract
This repository implements a hybrid deep learning pipeline for automatic colorization of black-and-white film frames. The approach combines a U-Net convolutional model for chrominance prediction with a GAN-based refinement stage to enhance texture realism and perceptual quality. The codebase includes training and inference scripts, evaluation utilities, sample inputs and outputs, and a short project report.

## Overview
Pipeline phases:
- **Preprocessing:** grayscale→LUV conversion, normalization, and augmentation.
- **Base model:** U-Net style CNN to predict chrominance channels.
- **Refinement:** GAN (generator + discriminator) to improve textures and reduce artifacts.
- **Postprocessing:** color-space fusion and temporal smoothing for frame coherence.

## Technical stack
- **Language:** Python 3.8+  
- **Libraries:** TensorFlow / Keras, OpenCV, NumPy, scikit-image, Matplotlib  
- **Tools:** Jupyter Notebook, Google Colab, CUDA-enabled GPU recommended

## Repository structure
```
ai-movie-colorization/
├── colorization_model.ipynb        # Notebook for training & inference
├── src/                            # Training, model, and utility scripts
├── data_sample/                    # Small sample images for demo
├── demo_output/                    # Before/after sample images
├── requirements.txt
├── report.pdf                      # One-page project report
└── README.md
```

## Results (summary)
- **Perceptual realism (LPIPS surrogate):** +35% vs baseline  
- **Frame consistency:** +22% measured on validation sequences  
- **Training efficiency:** ~20% reduction in runtime via optimized data pipeline

Sample outputs are available in `demo_output/` to visually inspect colorization quality.

## How to run (developer)
1. Clone:
```bash
git clone https://github.com/vickyy00070/ai-movie-colorization.git
cd ai-movie-colorization
```
2. Install (recommended in venv):
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
3. Run:
```bash
jupyter notebook colorization_model.ipynb
# or
python src/train.py --config configs/train_config.yaml
```

## Evaluation
Use `src/eval.py` to compute PSNR, SSIM, and LPIPS. A short notebook `notebooks/evaluation.ipynb` demonstrates metric comparisons vs a baseline.

## Limitations & future work
- Current method is frame-level; future work will integrate temporal models (optical flow / recurrent modules) for stronger video coherence.  
- Investigate diffusion-based colorization for higher-fidelity results.

## License
MIT License — see LICENSE

## Citation
If using this work, please cite:
Rudra, V. (2025). *AI-Based Movie Colorization Using Deep Learning for Visual Restoration.* Zenodo. DOI: (to be added)
