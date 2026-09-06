# FaceVerificationApp

Foundation Program project — a face verification application.

## Overview

This repository hosts the code, experiments, and documentation for the Face Verification App built as part of the Foundation project.

The use case: when a customer orders a high-value item (laptop, gold, etc.) through a delivery app, the delivery agent should be able to verify that the person receiving the package is actually the person who placed the order — before handing it over. Given two face images, decide whether they belong to the same person.

## Experiments

- **Siamese CNN (from scratch)** — a twin CNN encoder + L1 distance + sigmoid, trained from scratch on an LFW-style dataset per Koch et al. (2015). Reaches ~76% accuracy but falls short of the <1% FAR needed for production fraud detection; a data-scale limitation, not an architecture flaw. See [notebooks/siamese_face_verification.ipynb](notebooks/siamese_face_verification.ipynb) and the full write-up in [docs/siamese-model-writeup.md](docs/siamese-model-writeup.md).

## Project Structure

```
FaceVerificationApp/
├── src/                # Application source code
├── data/               # Datasets (not committed — see .gitignore)
├── notebooks/          # Jupyter notebooks for experimentation
├── models/             # Trained/saved models (not committed)
├── tests/              # Unit and integration tests
├── docs/               # Project documentation
├── requirements.txt    # Python dependencies
└── README.md
```

## Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/rkyadav915/FaceVerificationApp.git
   cd FaceVerificationApp
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Team

- Add teammates here.

## License

TBD
