# WMCL-HN: A Hybrid Network for Alzheimer's Disease Diagnosis

WMCL-HN (Weighted Modality Contrastive Learning Hybrid Network) is a multimodal deep learning framework designed for Alzheimer's Disease (AD) diagnosis. It combines feature learning and classifier learning to achieve efficient alignment and classification of multimodal data using contrastive learning and dynamic weight adjustment techniques.

## Features
- **Multimodal Feature Alignment**: Optimized alignment through specially designed contrastive loss functions:
  - **Intra-modal Alignment Loss**: Enhances the discriminative power within individual modalities.
  - **Cross-modal Alignment Loss**: Encourages consistency across different modalities.
- **Dynamic Weight Adjustment**: Automatically adjusts alignment weights based on the importance of each modality.
- **Hybrid Network Architecture**: Combines feature learning and classifier learning with a hierarchical training strategy.
- **Curriculum Learning Strategy**: Gradually transitions from feature learning to classifier optimization, improving model robustness and generalization.

## Data Requirements
The WMCL-HN model supports the following modalities:
- **MRI** (Magnetic Resonance Imaging)
- **PET** (Positron Emission Tomography)
- **CSF** (Cerebrospinal Fluid biomarkers)

## Installation
Ensure the following dependencies are installed:
- Python 3.8+
- PyTorch 1.10+
- NumPy
- pandas
- scikit-learn
- Matplotlib

Install required packages with:
```bash
pip install -r requirements.txt
