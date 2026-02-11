# SG-KRL

## Overview
This repository contains the official implementation of [IJCNN2025]"Fine-Grained Visual Classification Method Based on Semantically Guided Key Region Localization".

For questions or issues, please open an issue in this repository.

## Abstract
Fine-grained visual classification aims to distinguish visually similar categories and plays a critical role in practical applications. However, it faces challenges such as difficulty in localizing key regions and interference from background noise. Most methods fail to exploit semantic information effectively. This work propose a fine-grained classification method based on Semantically Guided Key Region Localization(SG-KRL). We first use deep global semantic information to guide feature fusion, enabling precise localization and adaptive cropping of the primary object region. Then, we apply semantic weighting in both channel and spatial dimensions to enhance object features, followed by multi-scale window selection to identify key parts. Finally, a graph-based representation is constructed using the semantic relationships of window positions, ensuring reliable classification. Experiments on three public benchmarks show our method balances accuracy and efficiency, outperforming existing methods. Additionally, we construct a luxury goods dataset containing genuine and counterfeit samples. Experiments on this dataset show high precision, demonstrating the method’s effectiveness in identifying subtle craftsmanship differences. Therefore, it can also contribute to intellectual property protection and combating illicit trade.

**Details**: [Read our paper](https://ieeexplore.ieee.org/document/11229252) for theoretical foundations and experimental results.  

## Supported Datasets
The code supports multiple fine-grained classification datasets (see `datasets.py` for details). Users are encouraged to test the method on additional public/private datasets.

## Environment Setup
- Python 3.9
- PyTorch 1.13

## Configuration
All training parameters can be modified in `config.py`.

## Training
The code supports single-machine multi-GPU parallel training.
```bash
python train.py
