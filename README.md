# [ICASSP 2026] Confidence-Aware Adaptive Semi-Supervised Approach for Reliable EEG-Based Emotion Recognition
Official implementation of  _Songha Kim, Soyeon Bak, Jun-Mo Kim, Woohyeok Choi, Yebin Choi and Tae-Eui Kam_, "Confidence-Aware Adaptive Semi-Supervised Approach for Reliable EEG-Based Emotion Recognition", 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Barcelona, Spain, May 4-8, 2026.
<p align="center">
  <img src="./CASPER_overview.png" alt="Main Overview" width="100%" />
</p>

## 🔍 TL;DR
- Confidence-aware pseudo-label adjustment with a two-stage strategy to reduce confirmation bias and improve reliability  
- Distance-weighted prototype learning to enhance robustness against noise and outliers  

## 📄 Paper
[CASPER Paper link](https://ieeexplore.ieee.org/document/11461997)

## 📦 Dataset Preparation
This project uses the [SEED](https://bcmi.sjtu.edu.cn/~seed/index.html) and [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/index.html) datasets. Please download and prepare them before running the code.

For the SEED dataset, DE (Differential Entropy) features can be extracted using [`data_prepare_seed.py`](./data_prepare_seed.py).

## 🛠️ Training 
To run CASPER, execute the main training script: [`implementation_CASPER.py`](./implementation_CASPER.py) 
This script handles the entire pipeline, including data preprocessing, semi-supervised training, and cross-subject evaluation.

The model architecture is defined in [`model_CASPER.py`](./model_CASPER.py).

We typically run the code as follows:
```bash
CUDA_VISIBLE_DEVICES=0 python implementation_CASPER.py --gpu=0
```

## 💘 Acknowledgements
The implementation code is bulit on the [EEGMatch](https://github.com/KAZABANA/EEGMatch) code base 

This work was supported by the Institute of Information and Communications Technology Planning and Evaluation (IITP) grant funded by the Korea government (MSIT) under the Artificial Intelligence Graduate School Program at Korea University (No. RS-2019-II190079), the Artificial Intelligence Research Hub Project (No. RS-2024-00457882), and Development of AI Autonomy and Knowledge Enhancement for AI Agent Collaboration (No. 2022-0-00871). This work was also supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No.RS-2023-00212498)
