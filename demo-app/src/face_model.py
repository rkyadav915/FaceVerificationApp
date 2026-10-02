"""
Face detection + embedding wrapper around facenet-pytorch.

Matches the architecture diagram's "Face detection & embedding (MTCNN + FaceNet)" box:
  - MTCNN finds and crops the face from a raw photo
  - InceptionResnetV1 (pretrained on VGGFace2) turns the cropped face into a
    512-dim embedding vector
  - Euclidean distance between two embeddings is the match score; smaller = more similar

This is the same pretrained-embeddings approach used in the final-review notebook,
wrapped for reuse by the app instead of re-run per notebook cell.
"""

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Threshold tiers locked on validation (see Slide 7 of the final deck).
# Distance < threshold => predicted match.
THRESHOLD_TIERS = {
    "High-value / age-restricted": 1.00,   # FAR 0.25%, FRR 19.4%
    "Standard": 1.05,                      # FAR 0.74%, FRR 14.0% (locked default)
    "Low-value / convenience": 1.20,       # FAR 7.35%, FRR 3.7%
}
DEFAULT_TIER = "Standard"


class FaceVerifier:
    """Loads MTCNN + FaceNet once and reuses them across requests."""

    def __init__(self):
        # keep_all=False: we expect exactly one face per photo (the recipient).
        self.mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=DEVICE)
        self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

    def get_embedding(self, pil_image: Image.Image) -> np.ndarray | None:
        """Detect the face in a PIL image and return its 512-d embedding.

        Returns None if no face was detected (caller should treat this as a
        hard failure, not a mismatch -- distinct from "different person").
        """
        face = self.mtcnn(pil_image.convert("RGB"))
        if face is None:
            return None
        with torch.no_grad():
            embedding = self.embedder(face.unsqueeze(0).to(DEVICE))
        return embedding.squeeze(0).cpu().numpy()

    @staticmethod
    def distance(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Euclidean distance between two embeddings (same metric as the notebook)."""
        return float(np.linalg.norm(embedding_a - embedding_b))

    def verify(self, embedding_a: np.ndarray, embedding_b: np.ndarray, tier: str = DEFAULT_TIER):
        """Compare two embeddings at a given risk tier's threshold.

        Returns (decision, distance, threshold) where decision is one of
        "accept", "step_up" (reject -> fallback to OTP/manual review).
        """
        threshold = THRESHOLD_TIERS[tier]
        dist = self.distance(embedding_a, embedding_b)
        decision = "accept" if dist < threshold else "step_up"
        return decision, dist, threshold
