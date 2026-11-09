import os

_host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11435")
OLLAMA_URL = _host if _host.startswith("http") else f"http://{_host}"

VISION_MODEL = os.environ.get("VISION_MODEL", "qwen3-vl:4b-instruct")
REPORT_MODEL = os.environ.get("REPORT_MODEL", VISION_MODEL)

NUM_CTX = int(os.environ.get("LEAF_NUM_CTX", "12288"))
FULL_IMAGE_SIDE = 1024      # whole photo sent to the model
CROP_IMAGE_SIDE = 768       # each zoomed close-up (cut from the original full-resolution photo)
SEARCH_RESULTS_PER_QUERY = 4