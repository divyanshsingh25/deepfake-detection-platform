"""
app.py  —  DeepShield
Dark forensic UI inspired by the reference design.
Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tempfile, time
from datetime import datetime
from pathlib import Path

import cv2, numpy as np, torch, torch.nn.functional as F
from PIL import Image
import streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.preprocessing   import get_inference_transforms, IMG_SIZE
from utils.face_extraction import get_face_detector, extract_face
from utils.gradcam         import get_target_layer, GradCAM
from utils.voting          import ensemble_vote, compute_risk_level, get_recommendation
from utils.report_generator import generate_report
from train import get_model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepShield — Forensic Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH_RESNET = "models/deepfake_model.pth"
MODEL_PATH_EFFNET = "models/deepfake_model_efficientnet.pth"
TEMP_DIR          = tempfile.mkdtemp()
FRAMES_PER_VIDEO  = 20
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# ── Global CSS  ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* Force dark background on everything */
.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #0f1535 50%, #0a0e27 100%) !important;
    min-height: 100vh;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }

/* ── Navbar ── */
.ds-navbar {
    background: rgba(2, 6, 23, 0.97);
    backdrop-filter: blur(12px);
    padding: 0.9rem 2.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(37,99,235,0.3);
    position: sticky; top: 0; z-index: 999;
}
.ds-logo { display:flex; align-items:center; gap:0.9rem; }
.ds-logo-icon {
    width:42px; height:42px;
    border:2px solid #2563eb; border-radius:8px;
    display:flex; align-items:center; justify-content:center;
    background:rgba(37,99,235,0.12);
    font-size:1.4rem; line-height:1;
}
.ds-logo-title { font-size:1.35rem; font-weight:700; letter-spacing:2px; color:#fff; margin:0; }
.ds-logo-sub   { font-size:0.72rem; color:#94a3b8; letter-spacing:1px; margin:0; }
.ds-nav-links  { display:flex; gap:2rem; list-style:none; margin:0; padding:0; }
.ds-nav-links a { color:#94a3b8; text-decoration:none; font-size:0.88rem; transition:color 0.2s; cursor:pointer; }
.ds-nav-links a:hover { color:#2563eb; }
.ds-status { display:flex; align-items:center; gap:0.5rem; color:#22c55e; font-size:0.82rem; font-weight:600; letter-spacing:1px; }
.ds-pulse {
    width:8px; height:8px; background:#22c55e; border-radius:50%;
    animation: dspulse 2s infinite;
}
@keyframes dspulse { 0%,100%{opacity:1} 50%{opacity:0.25} }

/* ── Privacy banner ── */
.ds-banner {
    display:flex; align-items:center; justify-content:center; gap:1.5rem;
    margin: 2rem 2.5rem 0;
    padding: 0.65rem 1rem;
    background: rgba(37,99,235,0.08);
    border: 1px solid rgba(37,99,235,0.25);
    border-radius: 8px;
    color: #94a3b8; font-size: 0.82rem;
}
.ds-banner span { color:#94a3b8; }
.ds-banner .sep { color:rgba(148,163,184,0.3); }

/* ── Main grid ── */
.ds-grid {
    display: grid;
    grid-template-columns: 1.15fr 1fr;
    gap: 1.75rem;
    padding: 1.75rem 2.5rem 2.5rem;
}

/* ── Upload card ── */
.ds-upload-card {
    background: rgba(2,6,23,0.65);
    border: 2px dashed rgba(148,163,184,0.25);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    transition: border-color 0.3s, background 0.3s;
}
.ds-upload-card:hover {
    border-color: rgba(37,99,235,0.5);
    background: rgba(2,6,23,0.85);
}
.ds-upload-icon {
    width:72px; height:72px; margin:0 auto 1.25rem;
    background:rgba(37,99,235,0.14); border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:2rem;
}
.ds-upload-title { font-size:1.6rem; font-weight:700; color:#fff; margin-bottom:0.35rem; }
.ds-upload-sub   { color:#94a3b8; font-size:0.9rem; margin-bottom:1.75rem; }

/* ── Right panel states ── */
.ds-await-panel {
    background: rgba(2,6,23,0.65);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    display: flex; flex-direction:column; align-items:center; justify-content:center;
    min-height: 360px;
}
.ds-await-icon { font-size:3.5rem; margin-bottom:1.25rem; opacity:0.45; }
.ds-await-title { font-size:1.4rem; font-weight:700; letter-spacing:2px; color:#64748b; margin-bottom:0.5rem; }
.ds-await-sub   { color:#475569; font-size:0.88rem; }

/* ── Terminal / scanning panel ── */
.ds-scan-panel {
    background: rgba(2,6,23,0.65);
    border: 1px solid rgba(37,99,235,0.3);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    min-height: 360px;
    display:flex; flex-direction:column; align-items:center; gap:1.5rem;
}
.ds-spinner {
    width:64px; height:64px; border-radius:50%;
    border:4px solid rgba(37,99,235,0.15);
    border-top:4px solid #2563eb;
    border-right:4px solid rgba(37,99,235,0.45);
    animation: dsspin 1.1s linear infinite;
}
@keyframes dsspin { to{transform:rotate(360deg)} }
.ds-scan-title { font-size:1.2rem; font-weight:700; letter-spacing:3px; color:#2563eb; text-transform:uppercase; }
.ds-terminal {
    width:100%; background:rgba(4,10,30,0.9);
    border:1px solid rgba(37,99,235,0.2); border-radius:10px;
    padding:1.1rem 1.4rem;
    font-family:'JetBrains Mono','Courier New',monospace;
    font-size:0.8rem; line-height:1.9; color:#22c55e;
    text-align:left; min-height:140px;
}

/* ── Result panel ── */
.ds-result-panel {
    background: rgba(2,6,23,0.85);
    border: 1px solid rgba(37,99,235,0.35);
    border-radius: 16px;
    padding: 2rem 2rem 1.5rem;
}
.ds-verdict-fake {
    background: linear-gradient(135deg,#7f1d1d,#991b1b);
    border:1px solid rgba(239,68,68,0.4);
    color:#fff; padding:1.1rem; border-radius:10px;
    text-align:center; font-size:1.65rem; font-weight:800;
    letter-spacing:2px; margin-bottom:1.25rem;
    box-shadow:0 6px 20px rgba(239,68,68,0.25);
}
.ds-verdict-real {
    background: linear-gradient(135deg,#14532d,#166534);
    border:1px solid rgba(34,197,94,0.4);
    color:#fff; padding:1.1rem; border-radius:10px;
    text-align:center; font-size:1.65rem; font-weight:800;
    letter-spacing:2px; margin-bottom:1.25rem;
    box-shadow:0 6px 20px rgba(34,197,94,0.25);
}
.ds-conf-row { display:flex; justify-content:space-around; margin:1rem 0; }
.ds-conf-item { text-align:center; }
.ds-conf-val-real { font-size:2.6rem; font-weight:800; color:#22c55e; line-height:1; }
.ds-conf-val-fake { font-size:2.6rem; font-weight:800; color:#ef4444; line-height:1; }
.ds-conf-lbl { font-size:0.78rem; color:#94a3b8; letter-spacing:1.5px; text-transform:uppercase; margin-top:0.35rem; }
.ds-pbar-track { width:100%; height:10px; background:rgba(30,41,59,0.8); border-radius:10px; overflow:hidden; margin:0.75rem 0 1.25rem; }
.ds-pbar-fill-real { height:100%; border-radius:10px; background:linear-gradient(90deg,#22c55e,#16a34a); transition:width 1s; }
.ds-pbar-fill-fake { height:100%; border-radius:10px; background:linear-gradient(90deg,#ef4444,#dc2626); transition:width 1s; }

/* ── Risk badge ── */
.ds-risk-high   { display:inline-block; background:#7f1d1d; color:#fca5a5; border:1px solid #ef4444; padding:0.25rem 0.9rem; border-radius:20px; font-size:0.8rem; font-weight:700; letter-spacing:1px; }
.ds-risk-medium { display:inline-block; background:#7c2d12; color:#fdba74; border:1px solid #f97316; padding:0.25rem 0.9rem; border-radius:20px; font-size:0.8rem; font-weight:700; letter-spacing:1px; }
.ds-risk-low    { display:inline-block; background:#14532d; color:#86efac; border:1px solid #22c55e; padding:0.25rem 0.9rem; border-radius:20px; font-size:0.8rem; font-weight:700; letter-spacing:1px; }

/* ── Analysis breakdown box ── */
.ds-breakdown {
    background:rgba(15,21,53,0.6); border:1px solid rgba(37,99,235,0.15);
    border-radius:10px; padding:1.1rem 1.25rem; margin-top:1rem;
}
.ds-breakdown h5 { color:#e2e8f0; font-size:0.88rem; margin-bottom:0.7rem; letter-spacing:0.5px; }
.ds-breakdown ul { list-style:none; padding:0; margin:0; }
.ds-breakdown li { color:#94a3b8; font-size:0.8rem; padding:0.3rem 0; display:flex; align-items:center; gap:0.5rem; }
.ds-breakdown li::before { content:"✓"; color:#2563eb; font-weight:800; }

/* ── Section cards (about / metrics) ── */
.ds-section { padding: 2rem 2.5rem; }
.ds-card {
    background: rgba(2,6,23,0.65);
    border: 1px solid rgba(37,99,235,0.2);
    border-radius: 14px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}
.ds-card h3 { color:#2563eb; font-size:1.3rem; margin-bottom:0.75rem; }
.ds-card p, .ds-card li { color:#94a3b8; line-height:1.8; font-size:0.9rem; }
.ds-feat-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:1rem; margin-top:1.25rem; }
.ds-feat-card { background:rgba(15,21,53,0.5); border:1px solid rgba(37,99,235,0.18); border-radius:10px; padding:1.1rem; }
.ds-feat-card h4 { color:#2563eb; margin-bottom:0.4rem; font-size:0.95rem; }
.ds-feat-card p  { font-size:0.82rem; color:#94a3b8; margin:0; }

/* ── Footer ── */
.ds-footer {
    text-align:center; padding:1.5rem 2rem;
    border-top:1px solid rgba(148,163,184,0.12);
    color:#475569; font-size:0.78rem;
}

/* ── Streamlit widget overrides ── */
.stFileUploader > div { background:transparent !important; border:none !important; }
.stFileUploader label { color:#94a3b8 !important; }

div[data-testid="stTabs"] button {
    background: transparent !important;
    color: #94a3b8 !important;
    border-bottom: 2px solid transparent !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1.25rem !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2563eb !important;
    border-bottom: 2px solid #2563eb !important;
}

/* Streamlit metric boxes – dark version */
div[data-testid="metric-container"] {
    background: rgba(15,21,53,0.7) !important;
    border: 1px solid rgba(37,99,235,0.2) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
}
div[data-testid="metric-container"] label { color:#94a3b8 !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color:#e2e8f0 !important; }

/* Progress bar color */
div[data-testid="stProgressBar"] > div > div { background-color: #2563eb !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg,#2563eb,#1d4ed8) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 20px rgba(37,99,235,0.35) !important; }

/* Download button */
.stDownloadButton > button {
    background: rgba(37,99,235,0.15) !important;
    color: #60a5fa !important;
    border: 1px solid rgba(37,99,235,0.4) !important;
    border-radius: 8px !important; font-weight:600 !important;
}

/* Expander */
details { background: rgba(15,21,53,0.5) !important; border: 1px solid rgba(37,99,235,0.2) !important; border-radius:10px !important; }
details summary { color: #94a3b8 !important; }

/* st.success / error / info */
div[data-testid="stAlert"] { border-radius:10px !important; }

/* Image captions */
.stImage figcaption { color:#64748b !important; font-size:0.78rem !important; }

/* Selectbox / slider labels */
label[data-testid="stWidgetLabel"] { color:#94a3b8 !important; }

/* Tab strip container */
div[data-testid="stTabs"] { background: transparent !important; padding: 0 2.5rem; border-bottom: 1px solid rgba(148,163,184,0.1); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cached loaders
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model_cached(model_name, model_path):
    if not os.path.exists(model_path):
        return None
    model = get_model(model_name, DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_detector():
    return get_face_detector(device="cpu")


# ══════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ══════════════════════════════════════════════════════════════════════════════
def predict_image(pil_image, model, transform, detector):
    img_rgb = np.array(pil_image.convert("RGB"))
    face_np = extract_face(img_rgb, detector)
    if face_np is None:
        face_np = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    face_pil = Image.fromarray(face_np)
    tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
    real_p, fake_p = float(probs[0]), float(probs[1])
    return (1 if fake_p > real_p else 0), real_p, fake_p, face_pil, tensor


def predict_video(video_path, model, transform, detector):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total-1, 0), FRAMES_PER_VIDEO, dtype=int)
    frame_results, face_images = [], []
    prog = st.progress(0, text="Scanning frames…")
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, bgr = cap.read()
        if not ret: continue
        rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        face_np = extract_face(rgb, detector)
        if face_np is None:
            face_np = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        fp      = Image.fromarray(face_np)
        face_images.append(fp)
        tensor  = transform(fp).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
        rp, fkp = float(probs[0]), float(probs[1])
        frame_results.append((1 if fkp > rp else 0, rp, fkp))
        prog.progress((i+1)/FRAMES_PER_VIDEO, text=f"Scanning frame {i+1}/{FRAMES_PER_VIDEO}…")
    cap.release(); prog.empty()
    return frame_results, face_images


def run_gradcam(model, model_name, face_pil, transform):
    try:
        model.train()  # train mode allows gradient flow through BatchNorm
        target_layer = get_target_layer(model, model_name)

        acts_store  = {}
        grads_store = {}

        def fwd(module, inp, out):
            acts_store["a"] = out

        def bwd(module, grad_in, grad_out):
            grads_store["g"] = grad_out[0]

        h1 = target_layer.register_forward_hook(fwd)
        h2 = target_layer.register_full_backward_hook(bwd)

        t = transform(face_pil).unsqueeze(0).to(DEVICE)

        with torch.enable_grad():
            logits = model(t)
            model.zero_grad()
            logits[0, 1].backward()  # backprop on "fake" class

        h1.remove()
        h2.remove()
        model.eval()

        acts  = acts_store.get("a")
        grads = grads_store.get("g")
        if acts is None or grads is None:
            return None

        weights = grads[0].mean(dim=(1, 2))
        cam_map = (weights[:, None, None] * acts[0]).sum(0)
        cam_map = torch.relu(cam_map).detach().cpu().numpy()
        if cam_map.max() > 0:
            cam_map = cam_map / cam_map.max()

        cam_resized = cv2.resize(cam_map, (224, 224))

        face_np = np.array(face_pil.resize((224, 224))).astype(np.float32) / 255.0
        face_u8 = (np.clip(face_np, 0, 1) * 255).astype(np.uint8)

        heat_u8  = (cam_resized * 255).astype(np.uint8)
        heat_col = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_col = cv2.cvtColor(heat_col, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(face_u8, 0.6, heat_col, 0.4, 0)
        return Image.fromarray(overlay)

    except Exception as e:
        try: model.eval()
        except: pass
        print("GradCAM error:", e)
        return None




# ══════════════════════════════════════════════════════════════════════════════
# NAVBAR  (rendered as HTML)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="ds-navbar">
  <div class="ds-logo">
    <div class="ds-logo-icon">🛡</div>
    <div>
      <p class="ds-logo-title">DEEPSHIELD</p>
      <p class="ds-logo-sub">Forensic Media Analysis System</p>
    </div>
  </div>

  <div class="ds-status">
    <div class="ds-pulse"></div>
    SYSTEM ONLINE
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_detect, tab_webcam, tab_metrics, tab_about = st.tabs([
    "🔍  Detect",
    "📷  Live Webcam",
    "📊  Model Metrics",
    "ℹ️  About & Ethics",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DETECT
# ══════════════════════════════════════════════════════════════════════════════
with tab_detect:

    # Privacy banner
    st.markdown("""
    <div class="ds-banner">
      <span>🔒 Local Processing</span><span class="sep">•</span>
      <span>🚫 No Cloud Storage</span><span class="sep">•</span>
      <span>🛡️ Privacy First</span><span class="sep">•</span>
      <span>⚡ Powered by ResNet50 / EfficientNet-B0</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Settings row (hidden in expander to keep clean look) ──────────────────
    with st.expander("⚙️  Settings", expanded=False):
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            model_choice = st.selectbox("Model", ["ResNet50", "EfficientNet-B0"])
        with scol2:
            threshold = st.slider("Fake threshold (%)", 50, 95, 60)
        with scol3:
            n_frames = st.slider("Video frames to sample", 5, 30, FRAMES_PER_VIDEO)

    model_name = "resnet50" if model_choice == "ResNet50" else "efficientnet"
    model_path = MODEL_PATH_RESNET if model_choice == "ResNet50" else MODEL_PATH_EFFNET

    # ── Two-column layout ──────────────────────────────────────────────────────
    st.markdown('<div class="ds-grid">', unsafe_allow_html=True)
    left_col, right_col = st.columns([1.15, 1], gap="large")

    # ── LEFT: Upload ───────────────────────────────────────────────────────────
    with left_col:
        st.markdown("""
        <style>
        /* Hide label */
        div[data-testid="stFileUploader"] > label { display:none !important; }

        /* Card shell around the uploader */
        div[data-testid="stFileUploader"] {
            background: rgba(2,6,23,0.65) !important;
            border: 2px dashed rgba(148,163,184,0.22) !important;
            border-radius: 16px !important;
            transition: border-color .3s !important;
        }
        div[data-testid="stFileUploader"]:hover {
            border-color: rgba(37,99,235,0.5) !important;
        }

        /* Dropzone area inside */
        div[data-testid="stFileUploaderDropzone"] {
            background: transparent !important;
            border: none !important;
            padding: 2.5rem 1.5rem 1.5rem !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            min-height: 240px !important;
            text-align: center !important;
            gap: 0.5rem !important;
        }

        /* Upload icon injected via CSS */
        div[data-testid="stFileUploaderDropzone"]::before {
            content: "⬆️";
            display: flex;
            align-items: center;
            justify-content: center;
            width: 64px; height: 64px;
            background: rgba(37,99,235,0.14);
            border-radius: 50%;
            font-size: 1.8rem;
            margin-bottom: 0.75rem;
        }

        /* Hide cloud SVG */
        div[data-testid="stFileUploaderDropzone"] svg { display:none !important; }

        /* "Drag and drop file here" text → title style */
        div[data-testid="stFileUploaderDropzone"] span {
            font-size: 1.35rem !important;
            font-weight: 700 !important;
            color: #ffffff !important;
        }

        /* "Limit 200MB" text → subtitle style */
        div[data-testid="stFileUploaderDropzone"] small {
            color: #64748b !important;
            font-size: 0.75rem !important;
        }

        /* Browse Files button */
        div[data-testid="stFileUploaderDropzone"] button {
            background: linear-gradient(135deg,#2563eb,#1d4ed8) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 8px !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            padding: 0.65rem 2.5rem !important;
            margin-top: 1rem !important;
            cursor: pointer !important;
            box-shadow: 0 4px 15px rgba(37,99,235,0.35) !important;
            transition: all 0.2s !important;
        }
        div[data-testid="stFileUploaderDropzone"] button:hover {
            background: linear-gradient(135deg,#1d4ed8,#1e40af) !important;
            box-shadow: 0 6px 22px rgba(37,99,235,0.5) !important;
            transform: translateY(-2px) !important;
        }

        /* File chip */
        div[data-testid="stFileUploaderFile"] {
            background: rgba(37,99,235,0.1) !important;
            border: 1px solid rgba(37,99,235,0.25) !important;
            border-radius: 8px !important;
            margin: 0 1rem 0.75rem !important;
        }
        div[data-testid="stFileUploaderFile"] span { color: #60a5fa !important; }
        </style>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "upload",
            type=["jpg","jpeg","png","mp4","avi","mov"],
            label_visibility="hidden",
            key="main_uploader",
        )

        if uploaded:
            suffix   = Path(uploaded.name).suffix.lower()
            is_video = suffix in [".mp4",".avi",".mov"]
            tmp_path = os.path.join(TEMP_DIR, uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            if is_video:
                st.video(tmp_path)
            else:
                # Fixed width — never full screen
                st.image(tmp_path, width=400)

    # ── RIGHT: Status / Result panel ──────────────────────────────────────────
    with right_col:

        if not uploaded:
            st.markdown("""
            <div class="ds-await-panel">
              <div class="ds-await-icon">⚡</div>
              <p class="ds-await-title">AWAITING INPUT</p>
              <p class="ds-await-sub">Upload media to begin forensic analysis</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Load model
            with st.spinner(""):
                model    = load_model_cached(model_name, model_path)
                detector = load_detector()
                transform= get_inference_transforms()

            if model is None:
                st.markdown("""
                <div class="ds-await-panel">
                  <div class="ds-await-icon">⚠️</div>
                  <p class="ds-await-title">MODEL NOT FOUND</p>
                  <p class="ds-await-sub">Run: python train.py --model resnet50 --data_dir dataset/split --epochs 15</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # ── SCANNING animation placeholder ─────────────────────────
                scan_ph = st.empty()
                scan_ph.markdown("""
                <div class="ds-scan-panel">
                  <div class="ds-spinner"></div>
                  <div class="ds-scan-title">Scanning Evidence</div>
                  <div class="ds-terminal">
                    &gt; Parsing bitstream headers...<br>
                    &gt; Extracting facial landmarks...<br>
                    &gt; Analysing GAN-generated noise...<br>
                    &gt; Calculating texture inconsistencies...<br>
                    &gt; Checking biological artifacts...<br>
                    &gt; Finalizing forensic verdict...
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Run inference ───────────────────────────────────────────
                if not is_video:
                    pil_image = Image.open(tmp_path).convert("RGB")
                    _, real_p, fake_p, face_pil, tensor = predict_image(
                        pil_image, model, transform, detector)
                    label      = "Fake" if fake_p*100 >= threshold else "Real"
                    confidence = fake_p*100 if label=="Fake" else real_p*100
                    risk       = compute_risk_level(label, confidence)
                    rec        = get_recommendation(label, risk)
                    real_count = fake_count = frame_count = 0
                    frame_results = []
                else:
                    frame_results, face_images = predict_video(
                        tmp_path, model, transform, detector)
                    result     = ensemble_vote(frame_results)
                    label      = result["label"]
                    confidence = result["confidence"]
                    real_count = sum(1 for c,_,_ in frame_results if c==0)
                    fake_count = len(frame_results)-real_count
                    frame_count= len(frame_results)
                    risk       = compute_risk_level(label, confidence)
                    rec        = get_recommendation(label, risk)
                    real_p     = result.get("avg_real_prob",0)/100
                    fake_p     = result.get("avg_fake_prob",0)/100
                    face_pil   = face_images[0] if face_images else None

                gradcam_img = run_gradcam(model, model_name, face_pil, transform) if face_pil else None

                # ── Replace scanner with result ─────────────────────────────
                scan_ph.empty()

                verdict_cls = "ds-verdict-fake" if label=="Fake" else "ds-verdict-real"
                icon        = "🚨" if label=="Fake" else "✅"
                bar_cls     = "ds-pbar-fill-fake" if label=="Fake" else "ds-pbar-fill-real"
                bar_w       = f"{confidence:.1f}"

                risk_cls  = {"High":"ds-risk-high","Medium":"ds-risk-medium"}.get(
                             risk.split("-")[0], "ds-risk-low")

                st.markdown(f"""
                <div class="ds-result-panel">
                  <div class="{verdict_cls}">{icon}&nbsp; {label.upper()} DETECTED</div>

                  <div class="ds-conf-row">
                    <div class="ds-conf-item">
                      <div class="ds-conf-val-real">{real_p*100:.1f}%</div>
                      <div class="ds-conf-lbl">Real</div>
                    </div>
                    <div class="ds-conf-item">
                      <div class="ds-conf-val-fake">{fake_p*100:.1f}%</div>
                      <div class="ds-conf-lbl">Fake</div>
                    </div>
                  </div>

                  <div class="ds-pbar-track">
                    <div class="{bar_cls}" style="width:{bar_w}%"></div>
                  </div>

                  <div style="text-align:center;margin-bottom:0.75rem;">
                    Risk Level &nbsp; <span class="{risk_cls}">{risk.upper()}</span>
                  </div>

                  <div class="ds-breakdown">
                    <h5>Forensic Analysis Breakdown</h5>
                    <ul>
                      <li>Frame-level facial texture analysis</li>
                      <li>Compression &amp; artifact detection</li>
                      <li>Confidence-weighted ensemble voting</li>
                      <li>Grad-CAM attention map generated</li>
                    </ul>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close ds-grid

    # ── Show results below grid (only when uploaded + analysed) ───────────────
    if uploaded and 'label' in dir() and model:

        st.markdown("<div style='padding:0 2.5rem'>", unsafe_allow_html=True)

        # Face + Grad-CAM
        if face_pil:
            st.markdown("---")
            gc1, gc2 = st.columns(2)
            with gc1:
                st.markdown("##### 🧑 Extracted Face Region")
                st.image(face_pil, width=280, caption="MTCNN face crop")
            with gc2:
                st.markdown("##### 🔥 Grad-CAM Attention Map")
                if gradcam_img:
                    st.image(gradcam_img, width=280,
                             caption="Warm regions = model focus areas")
                else:
                    if isinstance(gradcam_img, str):
                        st.error(f"Grad-CAM error: {gradcam_img}")
                    else:
                        st.info("Grad-CAM could not be generated.")

        # Video timeline
        if is_video and frame_results:
            st.markdown("---")
            st.markdown("##### 📈 Frame-level Fake Probability Timeline")
            fk_probs = [fp for _,_,fp in frame_results]
            fig, ax = plt.subplots(figsize=(11,3))
            fig.patch.set_facecolor("#0a0e27")
            ax.set_facecolor("#0f1535")
            ax.plot(fk_probs, color="#ef4444", lw=2, label="Fake prob")
            ax.axhline(threshold/100, color="#94a3b8", lw=1, ls="--",
                       label=f"Threshold {threshold}%")
            ax.fill_between(range(len(fk_probs)), fk_probs, threshold/100,
                            where=[fp>threshold/100 for fp in fk_probs],
                            alpha=0.25, color="#ef4444")
            ax.set_ylim(0,1); ax.set_xlabel("Frame", color="#94a3b8")
            ax.set_ylabel("Fake Probability", color="#94a3b8")
            ax.tick_params(colors="#64748b")
            for sp in ax.spines.values(): sp.set_color("#1e293b")
            ax.legend(facecolor="#0f1535", labelcolor="#94a3b8", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.markdown("##### 🧑 Sample Extracted Faces")
            cols = st.columns(min(5, len(face_images)))
            for i, col in enumerate(cols):
                if i < len(face_images):
                    fp_pct = frame_results[i][2]*100
                    col.image(face_images[i],
                              caption=f"F{i+1} | Fake:{fp_pct:.0f}%",
                              use_container_width=True)

        # Recommendation
        st.markdown("---")
        if label == "Fake":
            st.error(f"⚠️ **Recommendation:** {rec}")
        else:
            st.success(f"✅ **Recommendation:** {rec}")

        # PDF + Cyber portal
        st.markdown("---")
        pc1, pc2 = st.columns(2)

        with pc1:
            st.markdown("""
            <div style="background:rgba(37,99,235,0.08);border:1px solid rgba(37,99,235,0.25);
                        border-radius:10px;padding:1rem;margin-bottom:0.75rem;">
              <h5 style="color:#60a5fa;margin:0 0 0.4rem;">📄 Forensic PDF Report</h5>
              <p style="color:#94a3b8;font-size:0.82rem;margin:0;">
                Generate a full forensic report with metadata, Grad-CAM, and cybercrime portal link.
              </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("📥 Generate PDF Report", use_container_width=True):
                with st.spinner("Building report…"):
                    fp_path = hp_path = None
                    if face_pil:
                        fp_path = os.path.join(TEMP_DIR, "face_rpt.jpg")
                        face_pil.save(fp_path)
                    if gradcam_img:
                        hp_path = os.path.join(TEMP_DIR, "heat_rpt.jpg")
                        gradcam_img.save(hp_path)
                    rpt = os.path.join(TEMP_DIR, "deepshield_report.pdf")
                    generate_report(
                        output_path=rpt,
                        file_name=uploaded.name,
                        model_name=f"{'ResNet50' if model_name=='resnet50' else 'EfficientNet-B0'} (Transfer Learning)",
                        label=label, confidence=confidence, risk_level=risk,
                        recommendation=rec, frame_count=frame_count,
                        real_frames=real_count, fake_frames=fake_count,
                        heatmap_image_path=hp_path, face_image_path=fp_path,
                    )
                with open(rpt,"rb") as f:
                    st.download_button(
                        "⬇️ Download Report PDF", f.read(),
                        file_name=f"DeepShield_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf", use_container_width=True,
                    )

        with pc2:
            st.markdown("""
            <div style="background:rgba(127,29,29,0.15);border:1px solid rgba(239,68,68,0.3);
                        border-radius:10px;padding:1rem;margin-bottom:0.75rem;">
              <h5 style="color:#f87171;margin:0 0 0.4rem;">🚨 Report to Cyber Crime Portal</h5>
              <p style="color:#94a3b8;font-size:0.82rem;margin:0;">
                If this deepfake was used for fraud, harassment, or blackmail — report it now.
              </p>
            </div>
            """, unsafe_allow_html=True)
            st.link_button(
                "🔗 cybercrime.gov.in",
                "https://cybercrime.gov.in",
                use_container_width=True,
            )
            st.markdown(
                "<p style='text-align:center;color:#64748b;font-size:0.8rem;margin-top:0.4rem;'>"
                "📞 National Cyber Crime Helpline: <b style='color:#94a3b8'>1930</b></p>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — WEBCAM
# ══════════════════════════════════════════════════════════════════════════════
with tab_webcam:
    st.markdown('<div class="ds-section">', unsafe_allow_html=True)
    st.markdown("""
    <div class="ds-card">
      <h3>📷 Live Webcam Detection</h3>
      <p>Capture a photo with your webcam and run instant deepfake analysis on the detected face.</p>
    </div>
    """, unsafe_allow_html=True)

    w_model  = load_model_cached("resnet50", MODEL_PATH_RESNET)
    w_det    = load_detector()
    w_trans  = get_inference_transforms()
    snap     = st.camera_input("📸 Take a photo")

    if snap and w_model:
        pil = Image.open(snap).convert("RGB")
        with st.spinner("Analysing…"):
            _, rp, fp, face, _ = predict_image(pil, w_model, w_trans, w_det)
            lbl  = "Fake" if fp*100 >= 75 else "Real"
            conf = fp*100 if lbl=="Fake" else rp*100
            rsk  = compute_risk_level(lbl, conf)

        wc1, wc2, wc3 = st.columns(3)
        wc1.image(pil,  caption="Captured", use_container_width=True)
        wc2.image(face, caption="Face crop", use_container_width=True)
        with wc3:
            clr = "🔴" if lbl=="Fake" else "🟢"
            st.metric("Verdict",    f"{clr} {lbl}")
            st.metric("Confidence", f"{conf:.1f}%")
            st.metric("Risk",       rsk)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown('<div class="ds-section">', unsafe_allow_html=True)
    st.markdown('<div class="ds-card"><h3>📊 Model Performance Dashboard</h3></div>',
                unsafe_allow_html=True)

    paths = {
        "Training History"  : "models/deepfake_model_plot.png",
        "Confusion Matrix"  : "models/confusion_matrix.png",
        "ROC Curve"         : "models/roc_curve.png",
        "Model Comparison"  : "models/model_comparison.png",
    }
    existing = {k:v for k,v in paths.items() if os.path.exists(v)}

    if existing:
        keys = list(existing.keys())
        if "Training History" in existing:
            st.markdown("#### 📈 Training History")
            st.image(existing["Training History"], use_container_width=True)

        m1, m2 = st.columns(2)
        if "Confusion Matrix" in existing:
            m1.markdown("#### 🎯 Confusion Matrix")
            m1.image(existing["Confusion Matrix"])
        if "ROC Curve" in existing:
            m2.markdown("#### 📉 ROC Curve")
            m2.image(existing["ROC Curve"])
        if "Model Comparison" in existing:
            st.markdown("#### ⚖️ Model Comparison")
            st.image(existing["Model Comparison"], use_container_width=True)
    else:
        st.markdown("""
        <div class="ds-await-panel" style="min-height:220px;">
          <div class="ds-await-icon">📊</div>
          <p class="ds-await-title">NO METRICS YET</p>
          <p class="ds-await-sub">Run <code style='color:#60a5fa'>python evaluate.py</code> to generate charts</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown('<div class="ds-section">', unsafe_allow_html=True)

    st.markdown("""
    <div class="ds-card">
      <h3>🏗️ System Architecture</h3>
      <p>DeepShield is a fully local, modular AI pipeline with no paid APIs or cloud dependencies.</p>
      <ol style="color:#94a3b8;line-height:2;margin-left:1.5rem;font-size:0.9rem;">
        <li><b style="color:#60a5fa">Input Layer</b> — Images (JPG/PNG) or videos (MP4/AVI/MOV)</li>
        <li><b style="color:#60a5fa">Face Detection</b> — MTCNN with Haar Cascade fallback</li>
        <li><b style="color:#60a5fa">Preprocessing</b> — 224×224 resize, ImageNet normalisation, augmentation</li>
        <li><b style="color:#60a5fa">Backbone</b> — ResNet50 or EfficientNet-B0 (pretrained, fine-tuned)</li>
        <li><b style="color:#60a5fa">Classifier Head</b> — Dropout + FC → 2-class softmax</li>
        <li><b style="color:#60a5fa">Voting Engine</b> — Hard / Soft / Weighted / Ensemble (videos)</li>
        <li><b style="color:#60a5fa">Explainability</b> — Grad-CAM on final conv layer</li>
        <li><b style="color:#60a5fa">Report</b> — ReportLab forensic PDF with cybercrime portal link</li>
      </ol>
    </div>

    <div class="ds-card">
      <h3>⚙️ Key Features</h3>
      <div class="ds-feat-grid">
        <div class="ds-feat-card"><h4>🔒 Privacy-First</h4><p>All processing is local. Zero data sent to external servers.</p></div>
        <div class="ds-feat-card"><h4>🧠 Transfer Learning</h4><p>ImageNet-pretrained ResNet50 / EfficientNet-B0 fine-tuned for faces.</p></div>
        <div class="ds-feat-card"><h4>🔥 Grad-CAM</h4><p>Visual explanations showing which facial regions triggered the alert.</p></div>
        <div class="ds-feat-card"><h4>🎬 Video Analysis</h4><p>Frame sampling + ensemble voting for robust video-level verdict.</p></div>
        <div class="ds-feat-card"><h4>📄 PDF Reports</h4><p>Downloadable forensic reports with metadata, heatmaps, and recommendations.</p></div>
        <div class="ds-feat-card"><h4>🆓 100% Free</h4><p>PyTorch, OpenCV, Streamlit, ReportLab — all open-source.</p></div>
      </div>
    </div>

    <div class="ds-card">
      <h3>⚖️ Ethics & Responsible Use</h3>
      <ul style="color:#94a3b8;line-height:2;margin-left:1.5rem;font-size:0.9rem;">
        <li>This tool is for <b style="color:#60a5fa">detection only</b>, not creation of deepfakes</li>
        <li>Results are probabilistic — always verify with a qualified forensics expert before legal action</li>
        <li>No facial data is retained after the session ends</li>
        <li>Compliant with DPDP Act 2023 (India) data privacy principles</li>
        <li>Misuse to falsely accuse innocent individuals is a criminal offence</li>
      </ul>
    </div>

    <div class="ds-card">
      <h3>📚 Research Papers</h3>
      <ol style="color:#94a3b8;line-height:2.2;margin-left:1.5rem;font-size:0.88rem;">
        <li>Rössler et al. — <i>FaceForensics++</i>, ICCV 2019</li>
        <li>Afchar et al. — <i>MesoNet</i>, WIFS 2018</li>
        <li>Selvaraju et al. — <i>Grad-CAM</i>, ICCV 2017</li>
        <li>He et al. — <i>Deep Residual Learning (ResNet)</i>, CVPR 2016</li>
        <li>Tan &amp; Le — <i>EfficientNet</i>, ICML 2019</li>
        <li>Zhang et al. — <i>MTCNN</i>, IEEE SPL 2016</li>
        <li>Dolhansky et al. — <i>Deepfake Detection Challenge</i>, NeurIPS 2020</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ds-footer">
  <p>🛡 DeepShield v1.0 &nbsp;•&nbsp; AI-Based Deepfake Detection &nbsp;•&nbsp; Powered by PyTorch + Streamlit</p>
  <p>For verification purposes only &nbsp;•&nbsp; Not a replacement for human or legal judgment</p>
</div>
""", unsafe_allow_html=True)