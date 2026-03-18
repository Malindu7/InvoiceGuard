"""
InvoiceGuard — Streamlit Web UI
Member 5 — Integration Engineer

Run with:
    cd InvoiceGuard
    streamlit run app/app.py
"""

import os
import numpy as np
import streamlit as st
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from skimage.feature import hog
import io

# ─── CONFIG ──────────────────────────────────────────────────────────
IMG_SIZE   = 64
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.keras')
CLASSES    = ['Utilities', 'Travel', 'Office Supplies', 'Entertainment']
CLASS_ICONS = {'Utilities': '💡', 'Travel': '✈️', 'Office Supplies': '🖊️', 'Entertainment': '🎉'}
CLASS_COLORS = {
    'Utilities':       '#4CAF50',
    'Travel':          '#2196F3',
    'Office Supplies': '#FF9800',
    'Entertainment':   '#E91E63',
}


# ─── FEATURE EXTRACTION (must match Notebook 2) ──────────────────────
def extract_edge_density(flat_img, img_size=IMG_SIZE):
    img_u8 = (flat_img.reshape(img_size, img_size) * 255).astype(np.uint8)
    edges  = cv2.Canny(img_u8, 50, 150)
    return np.array([np.sum(edges > 0) / (img_size * img_size)], dtype=np.float32)


def extract_pixel_histogram(flat_img, n_bins=16):
    hist, _ = np.histogram(flat_img, bins=n_bins, range=(0, 1))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def extract_hog_features(flat_img, img_size=IMG_SIZE):
    img = flat_img.reshape(img_size, img_size)
    return hog(img, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), feature_vector=True).astype(np.float32)


def preprocess_image(pil_image):
    """Convert uploaded PIL image → feature vector ready for the model."""
    img_array = np.array(pil_image.convert('RGB'))
    gray      = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    resized   = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    flat       = normalized.flatten()

    edge  = extract_edge_density(flat)
    hist  = extract_pixel_histogram(flat)
    hog_f = extract_hog_features(flat)

    features = np.concatenate([flat, edge, hist, hog_f])
    return features.reshape(1, -1), normalized   # (1, n_features), 64×64 for display


# ─── LOAD MODEL ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)


# ─── PAGE SETUP ─────────────────────────────────────────────────────
st.set_page_config(
    page_title='InvoiceGuard',
    page_icon='🧾',
    layout='centered'
)

# ─── HEADER ─────────────────────────────────────────────────────────
st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px'>
        <h1 style='font-size:2.5rem; margin:0'>🧾 InvoiceGuard</h1>
        <p style='color:#666; font-size:1rem; margin-top:4px'>
            AI-powered Invoice Classification System
        </p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# ─── SIDEBAR ────────────────────────────────────────────────────────
with st.sidebar:
    st.header('ℹ️ About')
    st.markdown("""
    **InvoiceGuard** uses a trained Multi-Layer Perceptron (MLP) to automatically
    classify scanned invoice images into:

    - 💡 **Utilities** — electricity, water, internet
    - ✈️ **Travel** — hotels, flights, fuel
    - 🖊️ **Office Supplies** — stationery, furniture
    - 🎉 **Entertainment** — client meals, events

    **How it works:**
    1. Upload a scanned invoice
    2. The image is preprocessed (grayscale → 64×64 → normalized)
    3. Handcrafted features are extracted (edges, histograms, HOG)
    4. The MLP model predicts the category + confidence score
    """)

    st.divider()
    st.markdown('**Model:** MLP — 512→256→128→4')
    st.markdown('**Input:** 64×64 grayscale + feature engineered')

# ─── MAIN UI ────────────────────────────────────────────────────────
model = load_model()

if model is None:
    st.error(
        '⚠️ Model not found at `models/best_model.keras`.\n\n'
        'Please run **Notebook 3** first to train and save the model.',
        icon='🚨'
    )
    st.stop()

st.subheader('📤 Upload Invoice Image')
uploaded_file = st.file_uploader(
    'Choose a scanned invoice image (JPG, PNG, BMP)',
    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('**Original Image**')
        st.image(pil_image, use_column_width=True)

    # Preprocess & Predict
    with st.spinner('Analyzing invoice...'):
        features, processed_img = preprocess_image(pil_image)
        proba = model.predict(features, verbose=0)[0]
        pred_idx    = int(np.argmax(proba))
        pred_class  = CLASSES[pred_idx]
        confidence  = float(proba[pred_idx])

    with col2:
        st.markdown('**Preprocessed (64×64 grayscale)**')
        st.image(processed_img, use_column_width=True, clamp=True)

    st.divider()

    # ─── RESULT ──────────────────────────────────────────────────────
    color = CLASS_COLORS[pred_class]
    icon  = CLASS_ICONS[pred_class]

    st.markdown(f"""
        <div style='background:{color}22; border-left:5px solid {color};
                    padding:16px 20px; border-radius:8px; margin:10px 0'>
            <h2 style='margin:0; color:{color}'>{icon} {pred_class}</h2>
            <p style='margin:4px 0 0; color:#444; font-size:1rem'>
                Predicted Category
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Confidence gauge
    st.markdown('#### Confidence Score')
    st.progress(confidence)
    conf_label = '🟢 High' if confidence > 0.80 else ('🟡 Medium' if confidence > 0.55 else '🔴 Low')
    st.markdown(f'**{confidence*100:.1f}%** — {conf_label} confidence')

    if confidence < 0.55:
        st.warning(
            '⚠️ Low confidence prediction. This invoice may need manual review.',
            icon='⚠️'
        )

    # All class probabilities
    st.markdown('#### Probability Breakdown')
    for i, (cls, prob) in enumerate(zip(CLASSES, proba)):
        icon_cls = CLASS_ICONS[cls]
        col_a, col_b, col_c = st.columns([2, 5, 1])
        col_a.write(f'{icon_cls} **{cls}**')
        col_b.progress(float(prob))
        col_c.write(f'{prob*100:.1f}%')

    st.divider()

    # Export result
    result_text = (
        f"InvoiceGuard Classification Result\n"
        f"{'='*40}\n"
        f"File          : {uploaded_file.name}\n"
        f"Category      : {pred_class}\n"
        f"Confidence    : {confidence*100:.2f}%\n\n"
        f"All Probabilities:\n"
        + "\n".join([f"  {CLASS_ICONS[c]} {c:<20}: {p*100:.2f}%"
                     for c, p in zip(CLASSES, proba)])
    )

    st.download_button(
        label='📥 Download Classification Report',
        data=result_text,
        file_name=f'invoiceguard_{uploaded_file.name.split(".")[0]}.txt',
        mime='text/plain'
    )

else:
    # Placeholder prompt
    st.info('Upload an invoice image above to get started.', icon='📂')

    # Demo class cards
    st.markdown('#### Supported Invoice Categories')
    cols = st.columns(4)
    for col, (cls, icon) in zip(cols, CLASS_ICONS.items()):
        color = CLASS_COLORS[cls]
        col.markdown(f"""
            <div style='background:{color}22; border:1px solid {color};
                        padding:12px; border-radius:8px; text-align:center'>
                <div style='font-size:1.8rem'>{icon}</div>
                <div style='font-weight:bold; color:{color}; font-size:0.85rem'>{cls}</div>
            </div>
        """, unsafe_allow_html=True)
