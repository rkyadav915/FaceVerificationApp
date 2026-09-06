# Face Verification for Delivery Fraud Detection

A from-scratch Siamese neural network for face verification, built as part of a Foundation Project in applied machine learning.

## Overview

The idea: when a customer orders a high-value item (laptop, gold, etc.) through a delivery app, the delivery agent should be able to verify that the person receiving the package is actually the person who placed the order — before handing it over. This project builds and evaluates a face verification system for that use case: given two face images, decide whether they belong to the same person.

Rather than using pretrained face embeddings, this project trains a Siamese CNN entirely from scratch, following:

> Koch, G., Zemel, R., & Salakhutdinov, R. (2015). *Siamese Neural Networks for One-shot Image Recognition.* ICML Deep Learning Workshop. https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

## Approach

- **Architecture:** twin CNN encoders with shared weights → L1 distance between embeddings → sigmoid output (probability of match)
- **Training:** binary cross-entropy loss, trained on CPU with affine data augmentation (rotation, zoom, translation) to compensate for a low median image count per identity
- **Data:** LFW-style dataset, 1,322 identities / 7,233 images, used as a proxy for real in-app selfie/delivery-scan data
- **Pairs:** matching/mismatching pairs generated with a cap of 3 pairs per identity, to prevent a handful of over-photographed people from dominating training
- **Evaluation:** threshold selected on a validation set, then locked and applied unchanged to a held-out test set built from identities with zero overlap with training

## Results

| Metric | Validation (n=815) | Test (n=716, unseen identities) |
|---|---|---|
| Accuracy | 77.1% | 76.3% |
| Precision | 72.4% | 72.4% |
| Recall | 82.6% | 84.9% |
| FAR (False Accept Rate) | 33.3% | 32.4% |
| FRR (False Reject Rate) | 12.5% | 15.1% |

Test performance closely matches validation performance, despite the test identities never appearing during training — evidence the model learned a genuinely general similarity function rather than memorizing training identities.

## Limitations

The business target for this fraud-detection use case is **FAR < 1%**. This model's FAR (32.4%) falls well short of that, and pushing the decision threshold higher only reduces FAR to ~1.7% at the cost of an 88.5% False Reject Rate — not usable in practice.

This is a data-scale limitation rather than an architecture flaw: this project trained on ~5,400 pairs, while Koch et al. used 30,000–150,000+ pairs to reach 90%+ verification accuracy. For a production system, pretrained face embeddings (e.g. dlib ResNet, FaceNet) are the more practical path, since they're already trained on millions of faces.

## Project structure



## Running this project

1. Install dependencies: `torch`, `torchvision`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `Pillow`
2. Update the `data_dir` and `test_data_dir` paths in the notebook to point to your local dataset
3. Run cells in order — training is split into three phases (epochs 1–15, 16–30, 31–45) to allow checkpointing between runs
4. Trained model checkpoint is saved to `best_siamese_model.pt` (not included in this repo — see `.gitignore`)

**Note:** the dataset and trained model checkpoint are not included in this repository due to size. See the notebook for dataset structure expectations (one folder per identity).

## References

- Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese Neural Networks for One-shot Image Recognition. https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
- LFW dataset: http://vis-www.cs.umass.edu/lfw/
