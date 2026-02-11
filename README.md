# [IJCNN2025] SG-KRL

## Overview
This repository contains the official implementation of the fine-grained visual classification method described in our IJCNN2025 paper. The approach leverages semantically guided key region localization to improve classification performance.

## Supported Datasets
The code supports multiple fine-grained classification datasets (see `datasets.py` for details). Users are encouraged to test the method on additional public/private datasets.

## Environment Setup
- Python 3.9
- PyTorch 1.13

## Configuration
All training parameters can be modified in `config.py`.

## Training
```bash
python train.py

## Training Multi-GPU Training
The code supports single-machine multi-GPU parallel training.

## Contact
For questions or issues, please open an issue in this repository.

