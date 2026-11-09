import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import requests
import streamlit as st
from PIL import Image


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx(suppress_warning=True) is None:
        raise SystemExit(subprocess.call([
            sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
            "--server.address", "127.0.0.1", *sys.argv[1:],
        ]))

PROJECT_DIR = Path(__file__).resolve().parent
OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11435")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", f"127.0.0.1:{OLLAMA_PORT}")
OLLAMA_URL = OLLAMA_HOST if OLLAMA_HOST.startswith(("http://", "https://")) else f"http://{OLLAMA_HOST}"
os.environ["OLLAMA_HOST"] = OLLAMA_HOST
os.environ["OLLAMA_MODELS"] = str(PROJECT_DIR / "models")
os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
os.environ["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
os.environ["OLLAMA_KEEP_ALIVE"] = "30m"
os.environ["PYTHONUTF8"] = "1"
os.environ.setdefault("VISION_MODEL", "qwen3-vl:4b-instruct")
os.environ.setdefault("FALLBACK_MODELS", "qwen3-vl:4b gemma3:4b")
Path(os.environ["OLLAMA_MODELS"]).mkdir(parents=True, exist_ok=True)


def _ollama_executable():
    executable = shutil.which("ollama")
    if executable:
        return executable
    local_install = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    return str(local_install) if local_install.is_file() else None


def _ollama_ready():
    try:
        return requests.get(f"{OLLAMA_URL}/api/version", timeout=1.5).ok
    except requests.RequestException:
        return False


def _ollama_is_local():
    hostname = OLLAMA_HOST.split("://")[-1].split(":")[0].strip("[]").lower()
    return hostname in {"127.0.0.1", "localhost", "::1"}


def start_ollama():
    local = _ollama_is_local()
    executable = _ollama_executable() if local else None
    if local and not executable:
        webbrowser.open("https://ollama.com/download/windows")
        raise RuntimeError("Ollama is not installed. Install it, then restart Leaf Doctor.")

    if local and not _ollama_ready():
        subprocess.Popen(
            [executable, "serve"], cwd=str(PROJECT_DIR), env=os.environ.copy(),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    for _ in range(120 if not local else 40):
        if _ollama_ready():
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"Ollama is not reachable at {OLLAMA_URL}. Start the Ollama service and try again.")

    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
        installed = {item.get("name", "") for item in tags.get("models", [])}
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"Could not read Ollama models: {exc}") from exc

    candidates = [os.environ["VISION_MODEL"]] + os.environ["FALLBACK_MODELS"].split()
    selected = next((model for model in candidates if model in installed or f"{model}:latest" in installed), None)
    if selected is None:
        if local:
            for model in candidates:
                print(f"Downloading {model} into the project's models folder. The first download may take a while.")
                result = subprocess.run([executable, "pull", model], cwd=str(PROJECT_DIR), env=os.environ.copy())
                if result.returncode == 0:
                    selected = model
                    break
        else:
            raise RuntimeError(
                f"No configured vision model is installed on Ollama at {OLLAMA_URL}. "
                "Pull the model on the Ollama server and restart the app."
            )
    if selected is None:
        raise RuntimeError("Could not download a vision model. Check your internet connection and Ollama installation.")
    os.environ["VISION_MODEL"] = selected
    os.environ["REPORT_MODEL"] = selected
    return selected


@st.cache_resource(show_spinner="Starting local AI model…")
def initialize_ollama():
    return start_ollama()


st.set_page_config(page_title="Leaf Doctor", page_icon="🌿", layout="wide")
st.markdown("""
<style>
    .block-container {max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem;}
    .ld-hero {padding: 1.7rem 1.6rem 1.55rem; border-radius: 18px; margin-bottom: 1.25rem;
        background: linear-gradient(115deg, #123b2b 0%, #1f6847 58%, #39875a 100%); color: white;
        overflow: visible;}
    .ld-hero h1 {color: white; margin: 0 0 .45rem 0; font-size: 2.2rem; line-height: 1.25;}
    .ld-hero p {color: #e1f0e7; margin: 0; font-size: 1.05rem; line-height: 1.5;}
    .ld-hero .ld-kicker {display: block; color: #bce4c9; text-transform: uppercase;
        letter-spacing: .12em; font-size: .75rem; font-weight: 700; line-height: 1.6;
        padding: .12rem 0 .2rem; margin: 0 0 .45rem; overflow: visible;}
    div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}
    div[data-testid="stFileUploader"] {padding: .25rem 0;}
    .ld-muted {color: #64748b; font-size: .9rem;}
</style>
<div class="ld-hero">
  <div class="ld-kicker">Plant health assistant</div>
  <h1>Leaf Doctor</h1>
  <p>Upload a clear plant photo to review visible symptoms and get practical next steps.</p>
</div>
""", unsafe_allow_html=True)

try:
    selected_model = initialize_ollama()
except Exception as exc:
    st.error(str(exc))
    st.stop()

import config
from agent import check_ollama, run_agent

with st.sidebar:
    st.markdown("### About this analysis")
    st.caption("AI model running locally through Ollama")
    warning = check_ollama()
    if warning:
        st.warning(warning)
    st.divider()
    st.markdown("### Better photo tips")
    st.markdown(
        "- Use bright, even daylight\n"
        "- Keep the affected area in focus\n"
        "- Include the full leaf and a close-up when possible"
    )

input_col, result_col = st.columns([0.88, 1.35], gap="large")
with input_col:
    st.markdown("#### 1 · Add a plant photo")
    st.markdown('<p class="ld-muted">Choose an image or take one with your camera.</p>', unsafe_allow_html=True)
    upload_tab, camera_tab = st.tabs(["Upload", "Camera"])
    with upload_tab:
        uploaded_file = st.file_uploader(
            "JPG, PNG, or WEBP", type=["jpg", "jpeg", "png", "webp"], key="leaf_upload"
        )
    with camera_tab:
        camera_file = st.camera_input("Take a clear photo", key="leaf_camera")

    image_file = camera_file or uploaded_file
    if image_file:
        plant_image = Image.open(image_file).convert("RGB")
        st.image(plant_image, caption="Photo ready for analysis", use_container_width=True)
    else:
        st.info("Add a photo to start the plant health check.")

    analyze_clicked = st.button(
        "Analyze photo", type="primary", use_container_width=True, disabled=image_file is None
    )
    progress_container = st.empty()

with result_col:
    st.markdown("#### 2 · Analysis and recommendations")
    report_tab, evidence_tab = st.tabs(["Report", "Photo evidence"])
    with report_tab:
        report_area = st.empty()
        report_area.info("Your diagnosis and practical recommendations will appear here.")
    with evidence_tab:
        evidence_area = st.empty()
        evidence_area.caption("The areas used for closer symptom review will appear here.")

    if analyze_clicked and image_file:
        plant_image = Image.open(image_file).convert("RGB")
        with progress_container.container():
            with st.status("Reviewing your plant photo…", expanded=True) as analysis_status:
                progress_area = st.empty()
                has_report = False
                for progress, report, crops in run_agent(plant_image):
                    progress_area.markdown(progress)
                    if report:
                        report_area.markdown(report)
                        has_report = True
                    if crops:
                        with evidence_area.container():
                            st.markdown("##### Areas examined closely")
                            st.image(crops, use_container_width=True)
                analysis_status.update(label="Analysis complete", state="complete", expanded=False)
        if not has_report:
            report_area.warning("The analysis could not produce a report. Review the progress details and try again.")
