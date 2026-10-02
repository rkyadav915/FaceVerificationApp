"""
Face Verification for Fraud-Proof Delivery -- demo app.

Simulates the two "Client apps" from the architecture diagram inside one
Streamlit app (tabs), both talking to the same local FaceVerifier + store
that stand in for the "API service" / "Embedding store" / "Decision + audit
log" boxes:

  1. Shopping app  -> customer enrolls their face against an order ID
  2. Delivery app  -> agent photographs the collector; system verifies
  3. Audit log     -> ops view of every verification decision made

Run with:  streamlit run app.py
"""

import os
import sys

import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from face_model import DEFAULT_TIER, THRESHOLD_TIERS, FaceVerifier  # noqa: E402
from store import enroll_customer, get_enrollment, log_verification, read_audit_log  # noqa: E402

st.set_page_config(page_title="Face-Verified Delivery", page_icon="📦", layout="centered")


@st.cache_resource(show_spinner="Loading face-verification model (MTCNN + FaceNet)...")
def load_verifier() -> FaceVerifier:
    return FaceVerifier()


def get_photo(label: str, key: str):
    """Camera or file-upload input, returns a PIL Image or None."""
    source = st.radio(f"{label} — source", ["Camera", "Upload a file"], key=f"{key}_source", horizontal=True)
    if source == "Camera":
        shot = st.camera_input(label, key=f"{key}_camera")
    else:
        shot = st.file_uploader(label, type=["jpg", "jpeg", "png"], key=f"{key}_upload")
    if shot is None:
        return None
    return Image.open(shot)


st.title("📦 Face Verification for Fraud-Proof Delivery")
st.caption(
    '"Face verification that makes \'delivered\' mean \'delivered to the right person.\'" '
    "— Group 04, Foundation Project P3"
)

verifier = load_verifier()

tab_shop, tab_delivery, tab_audit = st.tabs(
    ["🛍️ Shopping App (Customer)", "🚚 Delivery App (Agent)", "📋 Audit Log (Ops)"]
)

# ---------------------------------------------------------------- #
# 1. Shopping app -- customer enrolls their face for an order
# ---------------------------------------------------------------- #
with tab_shop:
    st.subheader("Enroll your face for secure delivery")
    st.write(
        "When you place a high-value or age-restricted order, upload a clear photo of "
        "yourself. The delivery agent will check against this photo before handing over "
        "your parcel — no more shared OTPs or forged signatures."
    )

    order_id = st.text_input("Order ID", placeholder="e.g. ORD-10234", key="enroll_order_id")
    name = st.text_input("Your name", placeholder="e.g. Rama Krishna Yadav", key="enroll_name")
    photo = get_photo("Your photo", "enroll")

    if st.button("Enroll this photo", type="primary", disabled=not (order_id and name and photo)):
        with st.spinner("Detecting face and generating your verification profile..."):
            embedding = verifier.get_embedding(photo)
        if embedding is None:
            st.error("No face detected in that photo — please try a clearer, front-facing photo.")
        else:
            enroll_customer(order_id, name, embedding)
            st.success(f"Enrolled! Order **{order_id}** is now linked to {name}'s verified photo.")
            st.image(photo, caption="Reference photo on file", width=200)

# ---------------------------------------------------------------- #
# 2. Delivery app -- agent verifies the collector at the doorstep
# ---------------------------------------------------------------- #
with tab_delivery:
    st.subheader("Verify recipient at the doorstep")
    st.write("Enter the order ID and photograph the person collecting the parcel.")

    v_order_id = st.text_input("Order ID", placeholder="e.g. ORD-10234", key="verify_order_id")
    tier = st.selectbox(
        "Parcel risk tier",
        list(THRESHOLD_TIERS.keys()),
        index=list(THRESHOLD_TIERS.keys()).index(DEFAULT_TIER),
        help="Higher-risk parcels use a stricter (lower) distance threshold.",
    )
    st.caption(f"Threshold for this tier: **{THRESHOLD_TIERS[tier]}** (locked on validation)")
    collector_photo = get_photo("Photo of the collector", "verify")

    if st.button("Verify & decide", type="primary", disabled=not (v_order_id and collector_photo)):
        enrollment = get_enrollment(v_order_id)
        if enrollment is None:
            st.error(f"No enrollment found for order **{v_order_id}**. Ask the customer to enroll first.")
        else:
            with st.spinner("Detecting face and comparing against the reference photo..."):
                live_embedding = verifier.get_embedding(collector_photo)

            if live_embedding is None:
                st.error("No face detected in the collector's photo — please retake it.")
            else:
                decision, distance, threshold = verifier.verify(
                    enrollment["embedding"], live_embedding, tier=tier
                )
                log_verification(v_order_id, tier, distance, threshold, decision)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(collector_photo, caption="Collector (just captured)", width=200)
                with col2:
                    st.metric("Distance", f"{distance:.3f}", help=f"Threshold: {threshold}")

                if decision == "accept":
                    st.success(
                        f"✅ **ACCEPT** — matches enrolled recipient **{enrollment['name']}**. "
                        "Hand over the parcel and log the evidence."
                    )
                else:
                    st.warning(
                        "⚠️ **STEP UP** — does not clearly match the enrolled photo. "
                        "Do not hand over the parcel yet — fall back to OTP + manager review. "
                        "(Never a silent reject.)"
                    )

# ---------------------------------------------------------------- #
# 3. Audit log -- every verification decision, for ops monitoring
# ---------------------------------------------------------------- #
with tab_audit:
    st.subheader("Verification audit log")
    rows = read_audit_log()
    if not rows:
        st.info("No verifications logged yet — try the Delivery App tab first.")
    else:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        n_accept = sum(1 for r in rows if r["decision"] == "accept")
        n_step_up = len(rows) - n_accept
        c1, c2, c3 = st.columns(3)
        c1.metric("Total verifications", len(rows))
        c2.metric("Accepted", n_accept)
        c3.metric("Stepped up", n_step_up)

st.divider()
st.caption(
    "Demo app for Group 04's Foundation Project (P3 — Face Verification). "
    "Model: pretrained FaceNet (VGGFace2) via facenet-pytorch, locked threshold per risk tier. "
    "Not for production use — see the final report for limitations (no liveness detection, "
    "LFW-trained embeddings not validated on real doorstep captures)."
)
