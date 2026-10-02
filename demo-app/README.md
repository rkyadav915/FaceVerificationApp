# Face Verification for Fraud-Proof Delivery — Demo App

A Streamlit demo of Group 04's final-review model (pretrained FaceNet / VGGFace2, via
`facenet-pytorch`) wired into the delivery-verification flow from the architecture diagram:

```
Client apps (shopping + delivery)  ->  Face detection & embedding (MTCNN + FaceNet)
                                   ->  Embedding store (enroll / verify)
                                   ->  Distance + threshold (locked per risk tier)
                                   ->  Decision + audit log
```

This app simulates the two "client apps" as two tabs in one Streamlit UI so the whole
flow can be demoed/recorded from a single screen, without standing up a separate
frontend and API service.

## What it does

1. **🛍️ Shopping App (Customer)** — customer enters an Order ID + name and uploads/
   captures a reference photo. The app detects the face (MTCNN), computes a 512-d
   FaceNet embedding, and stores it keyed by Order ID.
2. **🚚 Delivery App (Agent)** — agent enters the same Order ID, picks a parcel risk
   tier, and captures/uploads a photo of the person at the door. The app computes that
   photo's embedding, measures the Euclidean distance to the enrolled embedding, and
   returns **Accept** (distance below the tier's threshold) or **Step Up** (fall back to
   OTP + manual review — never a silent reject, per the final deck's recommendation).
3. **📋 Audit Log (Ops)** — every verification attempt (order, tier, distance, decision,
   timestamp) is logged and viewable here.

Risk tiers and locked thresholds (from the final-review validation sweep, Slide 7):

| Parcel tier | Threshold | FAR | FRR |
|---|---|---|---|
| High-value / age-restricted | 1.00 | 0.25% | 19.4% |
| **Standard (default)** | **1.05** | 0.74% | 14.0% |
| Low-value / convenience | 1.20 | 7.35% | 3.7% |

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

First launch downloads the pretrained VGGFace2 weights (~107MB, one-time, cached by
`facenet-pytorch`). The app opens in your browser — the Camera inputs use your
webcam, or switch to "Upload a file" to use existing photos.

### Suggested demo flow (for the code-demo video)

1. Open the **Shopping App** tab, enter an order ID and your name, take a photo, enroll.
2. Switch to the **Delivery App** tab, enter the same order ID, take a second photo of
   yourself — should show **✅ ACCEPT**.
3. Take a photo of someone else (or use a different photo) with the same order ID —
   should show **⚠️ STEP UP**.
4. Open the **Audit Log** tab to show both attempts logged with their distances.

## Project structure

```
FaceVerificationApp/
├── app.py              Streamlit UI (3 tabs: enroll, verify, audit)
├── src/
│   ├── face_model.py   MTCNN + FaceNet wrapper (detect, embed, compare)
│   └── store.py        Local JSON embedding store + CSV audit log
├── data/               embeddings.json, audit_log.csv (created at runtime, gitignored)
└── requirements.txt
```

## Known limitations (carried over from the final report)

- Enrollment/verification photos are compared with **no liveness/anti-spoofing check** —
  a printed photo or screen replay would currently pass. Flagged as a next step in the
  final deck (Slide 10).
- Thresholds were tuned on **LFW** (web photos), not real doorstep captures — accuracy
  on actual delivery conditions (night lighting, motion blur, masks) is unvalidated.
- This demo's "embedding store" and "audit log" are local flat files, standing in for
  the diagram's embedding store / decision-log services — fine for a demo, not for
  production scale or multi-agent concurrent access.
- No fairness audit across demographics has been run yet (also flagged as a risk in
  the final report).
