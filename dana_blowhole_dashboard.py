import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(
    page_title="DANA | Blowhole Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# DANA BRAND CSS
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Barlow+Condensed:wght@500;600;700;800&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    font-family: 'Barlow', sans-serif !important;
    background: #f0f2f5 !important;
    color: #0d1e3a !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1e3a !important;
    border-right: 3px solid #2284c2 !important;
    min-width: 240px !important;
}
[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.85) !important;
    font-family: 'Barlow', sans-serif !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 1px;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(34,132,194,0.35) !important;
    margin: 0.6rem 0 !important;
}
[data-testid="stSidebar"] .stSuccess {
    background: rgba(30,138,72,0.25) !important;
    border-left: 3px solid #1e8a48 !important;
    border-radius: 3px !important;
    color: #7ee8a2 !important;
}
[data-testid="stSidebar"] .stError {
    background: rgba(217,52,42,0.25) !important;
    border-left: 3px solid #d9342a !important;
    border-radius: 3px !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #ffffff;
    border-bottom: 2px solid #d1d9e0;
    padding: 0 4px;
    gap: 0;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Barlow', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase !important;
    color: #78787a !important;
    padding: 10px 18px !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #2284c2 !important;
    border-bottom: 3px solid #2284c2 !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: #0d1e3a !important;
    background: #f8fafc !important;
}

/* ── Metric widget ── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #d1d9e0 !important;
    border-top: 3px solid #2284c2 !important;
    border-radius: 4px !important;
    padding: 14px 16px !important;
    box-shadow: none !important;
}
[data-testid="metric-container"] * {
    color: #0d1e3a !important;
}
[data-testid="metric-container"] label {
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: #78787a !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #0d1e3a !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] * {
    color: #0d1e3a !important;
}
[data-testid="metric-container"] p,
[data-testid="metric-container"] span,
[data-testid="metric-container"] div {
    color: #0d1e3a !important;
}

/* ── File uploader — force white text everywhere ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #2284c2 !important;
    border-radius: 4px !important;
    background: #f8fafc !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] * {
    font-family: 'Barlow', sans-serif !important;
}
/* The dark drag-and-drop inner box */
[data-testid="stFileUploaderDropzone"] {
    background: #1a2e4a !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #ffffff !important;
}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] div {
    color: #ffffff !important;
}
/* Browse files button */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] [data-testid="baseButton-secondary"] {
    background: #2284c2 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 3px !important;
}
[data-testid="stFileUploaderDropzone"] button * {
    color: #ffffff !important;
}
/* Fallback: any button inside uploader */
[data-testid="stFileUploader"] button {
    color: #ffffff !important;
    background: #2284c2 !important;
    border: none !important;
}
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button span {
    color: #ffffff !important;
}

/* ── Plotly chart ── */
[data-testid="stPlotlyChart"] {
    border: 1px solid #d1d9e0;
    border-radius: 4px;
    background: #ffffff;
}

/* ── Alert / info / success / error boxes ── */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    font-family: 'Barlow', sans-serif !important;
    font-size: 13px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] * {
    color: #2284c2 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f0f2f5; }
::-webkit-scrollbar-thumb { background: #c8ccd2; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2284c2; }

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }

/* ── Top-bar header ── */
.dana-topbar {
    background: #0d1e3a;
    border-bottom: 3px solid #2284c2;
    padding: 0 24px;
    height: 52px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -1rem 1.5rem -1rem;
}
.dana-logo {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 4px;
    color: #ffffff;
}
.dana-logo span { color: #4aa3d9; }
.dana-nav-right {
    display: flex;
    align-items: center;
    gap: 12px;
}
.dana-badge {
    font-family: 'Barlow', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    background: #2284c2;
    color: white;
    padding: 4px 10px;
    border-radius: 3px;
}
.dana-badge-live {
    background: #1e8a48;
}
.dana-nav-time {
    font-size: 11px;
    color: rgba(255,255,255,0.85) !important;
    font-family: 'Barlow Condensed', sans-serif;
    letter-spacing: 0.5px;
}

/* ── Page title block ── */
.dana-page-header {
    background: #ffffff;
    border: 1px solid #d1d9e0;
    border-left: 5px solid #2284c2;
    border-radius: 4px;
    padding: 16px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.dana-page-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 1px;
    color: #0d1e3a;
    text-transform: uppercase;
    margin: 0;
}
.dana-page-sub {
    font-size: 12px;
    color: #78787a;
    margin: 3px 0 0;
    font-weight: 500;
    letter-spacing: 0.3px;
}
.dana-breadcrumb {
    font-size: 11px;
    color: #78787a;
    text-align: right;
}
.dana-breadcrumb span { color: #2284c2; font-weight: 600; }

/* ── Section heading ── */
.dana-section-head {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #0d1e3a;
    border-bottom: 2px solid #2284c2;
    padding-bottom: 6px;
    margin: 0 0 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.dana-section-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #2284c2;
    flex-shrink: 0;
}

/* ── Upload placeholder ── */
.dana-upload-placeholder {
    background: #f8fafc;
    border: 2px dashed #2284c2;
    border-radius: 4px;
    padding: 48px 24px;
    text-align: center;
    color: #78787a;
}
.dana-upload-placeholder h3 {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #0d1e3a;
    margin-bottom: 8px;
}
.dana-upload-placeholder p { font-size: 12px; margin: 0; }

/* ── Result cards ── */
.dana-result-pass {
    background: #0d1e3a;
    border: 1px solid #1e8a48;
    border-left: 6px solid #1e8a48;
    border-radius: 4px;
    padding: 24px 20px;
    text-align: center;
    color: white;
}
.dana-result-fail {
    background: #0d1e3a;
    border: 1px solid #d9342a;
    border-left: 6px solid #d9342a;
    border-radius: 4px;
    padding: 24px 20px;
    text-align: center;
    color: white;
}
.dana-result-icon { font-size: 3rem; margin-bottom: 8px; }
.dana-result-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.dana-result-label.pass { color: #7ee8a2; }
.dana-result-label.fail { color: #f79590; }
.dana-result-conf {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
    color: rgba(255,255,255,0.7);
}
.dana-result-conf strong { color: #ffffff; font-size: 16px; }

/* ── Info panel ── */
.dana-info-panel {
    background: #e6f3fa;
    border: 1px solid #b5d4f4;
    border-left: 4px solid #2284c2;
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 12px;
    color: #0d1e3a;
    margin: 12px 0;
}
.dana-info-panel strong { font-weight: 700; }

/* ── Warning panel ── */
.dana-warn-panel {
    background: #fff8e6;
    border: 1px solid #f0c060;
    border-left: 4px solid #e08c1a;
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 12px;
    color: #7a5500;
    margin: 12px 0;
}

/* ── About table ── */
.dana-about-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin: 1rem 0;
}
.dana-about-table th {
    background: #0d1e3a;
    color: rgba(255,255,255,0.8);
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 9px 14px;
    text-align: left;
}
.dana-about-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #f0f2f5;
    color: #0d1e3a;
}
.dana-about-table tr:hover td { background: #f8fafc; }

/* ── Footer ── */
.dana-footer {
    background: #0d1e3a;
    border-top: 2px solid rgba(34,132,194,0.4);
    color: rgba(255,255,255,0.4);
    font-size: 10px;
    padding: 10px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 2rem -1rem -1rem -1rem;
    font-family: 'Barlow', sans-serif;
    letter-spacing: 0.3px;
}
.dana-footer-brand {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    letter-spacing: 2px;
    color: rgba(255,255,255,0.65);
    font-size: 12px;
}

/* ── Sidebar logo block ── */
.sidebar-logo-block {
    background: rgba(34,132,194,0.12);
    border: 1px solid rgba(34,132,194,0.3);
    border-radius: 4px;
    padding: 14px 12px;
    margin-bottom: 16px;
    text-align: center;
}
.sidebar-logo {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: 5px;
    color: #ffffff !important;
    line-height: 1;
}
.sidebar-logo-accent { color: #4aa3d9 !important; }
.sidebar-logo-sub {
    font-size: 9px;
    letter-spacing: 2px;
    color: rgba(255,255,255,0.45) !important;
    text-transform: uppercase;
    margin-top: 3px;
}

/* ── Sidebar stat row ── */
.sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    font-size: 12px;
}
.sidebar-stat-key { color: rgba(255,255,255,0.5) !important; font-size: 11px; }
.sidebar-stat-val { color: #ffffff !important; font-weight: 600; font-size: 12px; }

/* ── Live clock placeholder ── */
#dana-live-clock {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: #4aa3d9;
}
#dana-topbar-clock {
    font-size: 11px;
    color: rgba(255,255,255,0.85);
    font-family: 'Barlow Condensed', sans-serif;
    letter-spacing: 0.5px;
}
</style>

<script>
// Live clock updater — runs every second
(function startClocks() {
    function fmt(d) {
        const pad = n => String(n).padStart(2,'0');
        const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
        return {
            full: d.getDate() + ' ' + months[d.getMonth()] + ' ' + d.getFullYear()
                  + '  ' + pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds()),
            short: pad(d.getHours()) + ':' + pad(d.getMinutes())
                   + ' | ' + d.getDate() + ' ' + months[d.getMonth()] + ' ' + d.getFullYear()
        };
    }
    function tick() {
        const t = fmt(new Date());
        const sc = document.getElementById('dana-live-clock');
        const tc = document.getElementById('dana-topbar-clock');
        if (sc) sc.textContent = t.full;
        if (tc) tc.textContent = t.short;
    }
    tick();
    setInterval(tick, 1000);
})();
</script>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
CLASS_NAMES = ["Defective", "Non-defective"]
DEVICE = "cpu"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    try:
        model.load_state_dict(
            torch.load("blowhole_classifier.pth", map_location=DEVICE)
        )
        model.eval()
        return model, True
    except:
        return model, False

model, model_loaded = load_model()

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo-block">
        <div class="sidebar-logo">DANA<span class="sidebar-logo-accent">.</span></div>
        <div class="sidebar-logo-sub">Quality Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
              color:rgba(255,255,255,0.4) !important;margin:0 0 8px;">System Configuration</p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-stat"><span class="sidebar-stat-key">Model</span><span class="sidebar-stat-val">MobileNetV2</span></div>
    <div class="sidebar-stat"><span class="sidebar-stat-key">Task</span><span class="sidebar-stat-val">Blowhole Detection</span></div>
    <div class="sidebar-stat"><span class="sidebar-stat-key">Mode</span><span class="sidebar-stat-val">Image Inference</span></div>
    <div class="sidebar-stat"><span class="sidebar-stat-key">Device</span><span class="sidebar-stat-val">{DEVICE.upper()}</span></div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
              color:rgba(255,255,255,0.4) !important;margin:0 0 8px;">Deployment</p>
    <div class="sidebar-stat"><span class="sidebar-stat-key">Plant</span><span class="sidebar-stat-val">DANA Anand India</span></div>
    <div class="sidebar-stat"><span class="sidebar-stat-key">Division</span><span class="sidebar-stat-val">Automotive Casting</span></div>
    <div class="sidebar-stat"><span class="sidebar-stat-key">Project</span><span class="sidebar-stat-val">DIAPL</span></div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
              color:rgba(255,255,255,0.4) !important;margin:0 0 8px;">Model Status</p>
    """, unsafe_allow_html=True)

    if model_loaded:
        st.success("✅ Model Loaded Successfully")
    else:
        st.error("❌ Model File Not Found")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
              color:rgba(255,255,255,0.4) !important;margin:0 0 8px;">Workflow</p>
    <div style="font-size:12px;line-height:2;color:rgba(255,255,255,0.7) !important;">
        <div>① Upload casting image</div>
        <div>② AI model analyses defects</div>
        <div>③ Review result &amp; confidence</div>
        <div>④ Accept / Reject decision</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # SESSION TIME — uses live JS clock
    st.markdown("""
    <p style="font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
              color:rgba(255,255,255,0.4) !important;margin:0 0 6px;">Session Time</p>
    <div id="dana-live-clock" style="font-family:'Barlow Condensed',sans-serif;font-size:15px;
         font-weight:600;color:#4aa3d9 !important;margin:0;">--</div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# TOP BAR  (live clock via JS id)
# --------------------------------------------------
st.markdown(f"""
<div class="dana-topbar">
    <div style="display:flex;align-items:center;gap:16px;">
        <div class="dana-logo">DANA<span>.</span></div>
        <div style="width:1px;height:28px;background:rgba(255,255,255,0.15);"></div>
        <div style="font-size:11px;font-weight:500;letter-spacing:1.5px;
                    color:rgba(255,255,255,0.65);text-transform:uppercase;">
            Quality Intelligence Platform
        </div>
    </div>
    <div class="dana-nav-right">
        <span class="dana-badge dana-badge-live">&#x25CF;&nbsp;Live</span>
        <span class="dana-badge">DIAPL End</span>
        <span id="dana-topbar-clock" class="dana-nav-time">--</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------
st.markdown("""
<div class="dana-page-header">
    <div>
        <p class="dana-page-title">Blowhole Defect Detection System</p>
        <p class="dana-page-sub">AI-Powered Pre-Machining Quality Inspection &nbsp;·&nbsp; Precision Casting Components</p>
    </div>
    <div class="dana-breadcrumb">
        Manufacturing &nbsp;›&nbsp; DIAPL &nbsp;›&nbsp; <span>Blowhole Detection</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📸  Inspection", "📊  Analytics", "ℹ️  About"])

# ══════════════════════════════════════════
# TAB 1 — INSPECTION
# ══════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("""
        <div class="dana-section-head">
            <div class="dana-section-dot"></div>Upload Casting Image
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Select an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of the casting component for defect analysis"
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Component Under Inspection", use_container_width=True)

            st.markdown("""
            <div class="dana-section-head" style="margin-top:18px;">
                <div class="dana-section-dot"></div>Image Details
            </div>
            """, unsafe_allow_html=True)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Width", f"{image.size[0]} px")
            col_b.metric("Height", f"{image.size[1]} px")
            col_c.metric("Format", uploaded_file.type.split('/')[-1].upper())
        else:
            st.markdown("""
            <div class="dana-upload-placeholder">
                <h3>No Image Uploaded</h3>
                <p>Upload a casting image to begin quality inspection</p>
                <p style="margin-top:6px;color:#b0b8c4;">Supported: JPG &nbsp;·&nbsp; JPEG &nbsp;·&nbsp; PNG</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="dana-section-head">
            <div class="dana-section-dot"></div>Inspection Results
        </div>
        """, unsafe_allow_html=True)

        if uploaded_file and model_loaded:
            with st.spinner("Analysing component..."):
                img_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)

                label = CLASS_NAMES[pred.item()]
                confidence = conf.item() * 100

                if label == "Defective":
                    st.markdown(f"""
                    <div class="dana-result-fail">
                        <div class="dana-result-icon">⚠️</div>
                        <div class="dana-result-label fail">Defective Component</div>
                        <div class="dana-result-conf">
                            Detection Confidence &nbsp;—&nbsp; <strong>{confidence:.2f}%</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="dana-warn-panel">
                        <strong>Recommendation:</strong> Reject component — Send for scrap or rework.
                        Do not proceed to machining.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="dana-result-pass">
                        <div class="dana-result-icon">✅</div>
                        <div class="dana-result-label pass">Non-Defective</div>
                        <div class="dana-result-conf">
                            Detection Confidence &nbsp;—&nbsp; <strong>{confidence:.2f}%</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div class="dana-info-panel">
                        <strong>Recommendation:</strong> Accept component — Clear to proceed to machining.
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div class="dana-section-head" style="margin-top:20px;">
                    <div class="dana-section-dot"></div>Classification Probabilities
                </div>
                """, unsafe_allow_html=True)

                fig = go.Figure(data=[
                    go.Bar(
                        x=CLASS_NAMES,
                        y=[probs[0][i].item() * 100 for i in range(len(CLASS_NAMES))],
                        marker_color=["#d9342a", "#2284c2"],
                        marker_line_width=0,
                        text=[f"{probs[0][i].item()*100:.2f}%" for i in range(len(CLASS_NAMES))],
                        textposition="auto",
                        textfont=dict(family="Barlow, sans-serif", size=12, color="white"),
                    )
                ])
                fig.update_layout(
                    yaxis_title="Probability (%)",
                    xaxis_title="Classification",
                    showlegend=False,
                    height=260,
                    margin=dict(l=20, r=20, t=12, b=20),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(family="Barlow, sans-serif", color="#0d1e3a"),
                    yaxis=dict(gridcolor="#f0f2f5", tickfont=dict(size=11)),
                    xaxis=dict(tickfont=dict(size=12, color="#0d1e3a")),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="dana-section-head">
                    <div class="dana-section-dot"></div>Detailed Probabilities
                </div>
                """, unsafe_allow_html=True)
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.metric("Defective Probability", f"{probs[0][0].item()*100:.2f}%")
                with mc2:
                    st.metric("Non-defective Probability", f"{probs[0][1].item()*100:.2f}%")

        elif not model_loaded:
            st.error("⚠️ Model weights not found. Verify `blowhole_classifier.pth` path.")
        else:
            st.markdown("""
            <div class="dana-info-panel">
                Upload a casting image on the left to begin the inspection process.
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="dana-section-head">
        <div class="dana-section-dot"></div>System Analytics
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dana-info-panel">
        Analytics will display inspection history, defect rates, and performance metrics
        once the system is in production and accumulating session data.
    </div>
    """, unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Total Inspections", "—", help="Total inspections performed this session")
    with mc2:
        st.metric("Defect Rate", "—", help="Percentage of defective components detected")
    with mc3:
        st.metric("Model Accuracy", "94.5%", help="Classification accuracy on test dataset")
    with mc4:
        st.metric("Avg. Confidence", "—", help="Average prediction confidence")

# ══════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="dana-section-head">
        <div class="dana-section-dot"></div>System Overview
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("""
        <table class="dana-about-table">
            <thead><tr><th>Parameter</th><th>Detail</th></tr></thead>
            <tbody>
                <tr><td>Organisation</td><td>Dana Anand India Pvt. Ltd.</td></tr>
                <tr><td>Department</td><td>Automotive Casting Division</td></tr>
                <tr><td>Project</td><td>DIAPL Blowhole Detection</td></tr>
                <tr><td>Application</td><td>Pre-machining Quality Inspection</td></tr>
                <tr><td>Version</td><td>1.0</td></tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <table class="dana-about-table">
            <thead><tr><th>Parameter</th><th>Detail</th></tr></thead>
            <tbody>
                <tr><td>Model Architecture</td><td>MobileNetV2</td></tr>
                <tr><td>Classification Accuracy</td><td>94.5%+</td></tr>
                <tr><td>Inference Time</td><td>&lt; 1 second / image</td></tr>
                <tr><td>Inference Device</td><td>CPU</td></tr>
                <tr><td>Input Format</td><td>JPG, JPEG, PNG</td></tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dana-section-head" style="margin-top:20px;">
        <div class="dana-section-dot"></div>How It Works
    </div>
    <table class="dana-about-table">
        <thead><tr><th>Step</th><th>Action</th><th>Description</th></tr></thead>
        <tbody>
            <tr><td>01</td><td>Image Upload</td><td>Upload a high-quality image of the casting component (JPG/PNG).</td></tr>
            <tr><td>02</td><td>Pre-processing</td><td>Image is resized to 224×224 and normalised using ImageNet statistics.</td></tr>
            <tr><td>03</td><td>AI Inference</td><td>MobileNetV2 classifies the component as Defective or Non-defective.</td></tr>
            <tr><td>04</td><td>Decision Support</td><td>Result and confidence score are displayed with an Accept / Reject recommendation.</td></tr>
        </tbody>
    </table>

    <div class="dana-warn-panel" style="margin-top:16px;">
        <strong>Important:</strong> This system is intended for <strong>decision support</strong> only.
        Final quality decisions should involve additional inspection methods and trained QA personnel.
        Regular model retraining with production data is recommended.
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(f"""
<div class="dana-footer">
    <span class="dana-footer-brand">DANA INCORPORATED</span>
    <span>DIAPL Casting Quality Control System &nbsp;·&nbsp; Version 1.0 &nbsp;·&nbsp; Blowhole Detection</span>
    <span>For support, contact the Quality Assurance Department</span>
</div>
""", unsafe_allow_html=True)