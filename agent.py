import base64
import io
import json
import re

import numpy as np
import requests
from PIL import Image, ImageOps

import config
import kb


class OllamaError(Exception):
    pass


# ---------------------- Ollama helpers

def _b64(img: Image.Image, max_side: int) -> tuple[str, tuple[int, int]]:
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode(), img.size


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return re.sub(r"<think>.*$", "", text, flags=re.S).strip()


def _post(payload, stream=False):
    try:
        r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload, stream=stream, timeout=900)
    except requests.ConnectionError:
        raise OllamaError(f"Cannot reach Ollama at {config.OLLAMA_URL}. Start the app.")
    if r.status_code != 200:
        raise OllamaError(f"Ollama error {r.status_code}: {r.text[:300]}")
    return r


def _payload(model, messages, json_mode, temperature, stream):
    p = {"model": model, "messages": messages, "stream": stream, "keep_alive": "30m",
         "options": {"temperature": temperature, "num_ctx": config.NUM_CTX}}
    if json_mode:
        p["format"] = "json"
    return p


def chat(model, messages, json_mode=False, temperature=0.1) -> str:
    r = _post(_payload(model, messages, json_mode, temperature, False))
    return _strip_think(r.json().get("message", {}).get("content", ""))


def chat_stream(model, messages, temperature=0.3):
    r = _post(_payload(model, messages, False, temperature, True), stream=True)
    text = ""
    for line in r.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        text += chunk.get("message", {}).get("content", "")
        yield _strip_think(text)
        if chunk.get("done"):
            break


def _json(text: str) -> dict:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {}


def _vision(prompt: str, images_b64: list[str]) -> dict:
    msgs = [{"role": "user", "content": prompt, "images": images_b64}]
    data = _json(chat(config.VISION_MODEL, msgs, json_mode=True))
    if not data:  # one retry - small models sometimes break JSON
        data = _json(chat(config.VISION_MODEL, msgs, json_mode=True, temperature=0.3))
    return data


# ---------------------------------- image tools

def prepare(image: Image.Image) -> Image.Image:
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    return image.convert("RGB")


def quality_check(img: Image.Image) -> dict:
    small = img.copy()
    small.thumbnail((1024, 1024))
    g = np.asarray(small.convert("L"), dtype=np.float32)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    sharp, bright = float(lap.var()), float(g.mean())
    issues = []
    if min(img.size) < 400:
        issues.append("low resolution")
    if sharp < 40:
        issues.append("blurry")
    if bright < 45:
        issues.append("too dark")
    if bright > 225:
        issues.append("over-exposed")
    return {"sharpness": round(sharp), "brightness": round(bright), "issues": issues}


def _auto_regions(img: Image.Image, n=3):
    small = img.copy()
    small.thumbnail((384, 384))
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    green = (h >= 65) & (h <= 170) & (s > 0.15) & (v > 0.15)
    lesion = (((h < 65) | (h > 330)) & (s > 0.25) & (v > 0.2)) | ((s < 0.12) & (v > 0.75))

    
    r = 6
    g = np.pad(green.astype(np.float32), r, mode="edge")
    ii = np.pad(g.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    k = 2 * r + 1
    green_frac = (ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]) / (k * k)
    leaf_lesion = lesion & (green_frac > 0.6)
    H, W = green.shape
    cells = []
    for gy in range(3):
        for gx in range(3):
            y0, y1, x0, x1 = gy * H // 3, (gy + 1) * H // 3, gx * W // 3, (gx + 1) * W // 3
            score = leaf_lesion[y0:y1, x0:x1].mean() * min(green[y0:y1, x0:x1].mean() * 3, 1.0)
            cells.append((score, [x0 / W * 1000, y0 / H * 1000, x1 / W * 1000, y1 / H * 1000]))
    cells.sort(key=lambda c: c[0], reverse=True)
    return [b for sc, b in cells[:n] if sc > 0.005] or [[200, 200, 800, 800]]


def _parse_boxes(raw, sent_size):
    out = []
    for b in raw if isinstance(raw, list) else []:
        if isinstance(b, dict):
            b = b.get("box") or b.get("bbox") or b.get("bbox_2d") or list(b.values())
        try:
            x1, y1, x2, y2 = [float(v) for v in list(b)[:4]]
        except (TypeError, ValueError):
            continue
        mx = max(x1, y1, x2, y2)
        if mx <= 1.0:
            x1, y1, x2, y2 = x1 * 1000, y1 * 1000, x2 * 1000, y2 * 1000
        elif mx > 1000:
            W, H = sent_size
            x1, x2, y1, y2 = x1 / W * 1000, x2 / W * 1000, y1 / H * 1000, y2 / H * 1000
        x1, x2 = sorted((max(0, min(1000, x1)), max(0, min(1000, x2))))
        y1, y2 = sorted((max(0, min(1000, y1)), max(0, min(1000, y2))))
        if x2 - x1 >= 5 and y2 - y1 >= 5:
            out.append([x1, y1, x2, y2])
    return out[:3]


def _crop(img: Image.Image, box, min_frac=0.22, pad=0.2) -> Image.Image:
    W, H = img.size
    x1, y1, x2, y2 = box[0] / 1000 * W, box[1] / 1000 * H, box[2] / 1000 * W, box[3] / 1000 * H
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max((x2 - x1) * (1 + 2 * pad), (y2 - y1) * (1 + 2 * pad), min_frac * min(W, H))

    def axis(c, size, total):
        size = min(size, total)
        lo = min(max(c - size / 2, 0), total - size)
        return int(lo), int(lo + size)

    l, r = axis(cx, side, W)
    t, b = axis(cy, side, H)
    return img.crop((l, t, r, b))


# -------------------------------- agent steps
OBSERVE_PROMPT = """You are an expert plant pathologist examining a farmer's photo.
Do NOT diagnose yet. Return ONLY this JSON:
{
 "is_plant": true or false,
 "plant_common_name": "best guess, e.g. Maize",
 "plant_scientific_name": "e.g. Zea mays",
 "plant_id_confidence": 0-100,
 "plant_part": "leaf | fruit | stem | flower | root | whole plant",
 "looks_healthy": true or false,
 "symptoms": ["short list of visible problems"],
 "affected_regions": [[x1, y1, x2, y2], ...]
}
"affected_regions": up to 3 boxes around the MOST damaged spots/lesions, as coordinates normalised
0-1000 (0,0 = top-left, 1000,1000 = bottom-right). Empty list if healthy."""

DETAIL_PROMPT = """Image 1 is the whole photo of a {plant} {part}. Images 2+ are ZOOMED close-ups of the most affected areas.
Study the close-ups like a plant pathologist. Describe ONLY what is clearly visible; write "not visible" instead of guessing.
Return ONLY this JSON:
{{
 "spot_colour": "e.g. orange, cinnamon-brown, tan, grey centre with brown edge, yellow, black, white",
 "spot_shape": "round | oval | angular/rectangular bounded by veins | long cigar/boat | streaks/stripes | irregular blotch | winding tunnels | none",
 "spot_size": "pin-point <1 mm | small 1-5 mm | medium 5-20 mm | large >2 cm",
 "raised_pustules_or_powder": "no, or describe (e.g. raised orange powdery pustules)",
 "concentric_rings": true/false,
 "yellow_halo": true/false,
 "dark_margin": true/false,
 "water_soaked_or_greasy": true/false,
 "mould_or_fuzz": "none | white powder | grey/purple fuzz | black sooty | other",
 "tunnels_or_mines_inside_leaf": true/false,
 "holes_or_chewing": true/false,
 "insects_or_eggs": "describe exactly, or none",
 "mosaic_or_mottling": true/false,
 "curling_or_distortion": true/false,
 "pattern": "scattered | clustered | along veins | between veins | leaf edges | leaf tip | whole leaf",
 "leaf_age_affected": "older/lower | young/upper | all | unknown",
 "other_notes": "anything else important (e.g. dust, pollen, soil splash that is NOT disease)"
}}"""


def observe(full_b64):
    return _vision(OBSERVE_PROMPT, [full_b64])


def detail(images_b64, obs):
    prompt = DETAIL_PROMPT.format(plant=obs.get("plant_common_name", "plant"), part=obs.get("plant_part", "leaf"))
    return _vision(prompt, images_b64)


def diagnose(images_b64, obs, features, entries):
    reference = kb.as_text(entries + kb.GENERIC)
    prompt = f"""You are an expert plant pathologist doing a DIFFERENTIAL diagnosis.
Plant: {obs.get('plant_common_name')} ({obs.get('plant_scientific_name')}), part: {obs.get('plant_part')}.
Lesion features seen in the close-ups:
{json.dumps(features, ensure_ascii=False, indent=1)}

Known problems and their typical signs:
{reference}

Compare the features and the images with EACH known problem. A good match needs the KEY signs
(shape, colour, raised pustules vs flat spots, rings, halo, vein-bounded, tunnels, insects).
Do not choose a pest unless insects, eggs, mines or chewing are really visible.
Pick the 3 best matches (you may name a problem not in the list if it clearly fits better).
Return ONLY this JSON, sorted by confidence:
{{"candidates": [{{"name": "...", "cause": "...", "category": "fungal | bacterial | viral | pest | nutrient deficiency | environmental | healthy",
  "confidence": 0-100, "matching_signs": "...", "conflicting_signs": "..."}}]}}"""
    return _vision(prompt, images_b64).get("candidates", [])[:3]


def verify(images_b64, obs, candidates, entries):
    lines = []
    for c in candidates:
        signs = kb.signs_for(c.get("name"), entries + kb.GENERIC) or c.get("matching_signs", "")
        lines.append(f"- {c.get('name')} ({c.get('cause')}): typical signs = {signs}")
    prompt = f"""Second-opinion check on a {obs.get('plant_common_name')} {obs.get('plant_part')}.
Image 1 = whole photo, images 2+ = close-ups. Candidate diagnoses:
{chr(10).join(lines)}

Look at the close-ups again. For EACH candidate, check its typical signs one by one and decide if each is
PRESENT or ABSENT in the photo. Choose the candidate with the most key signs present and no strong conflict.
If two fit almost equally, choose one but lower the confidence.
Return ONLY this JSON:
{{"checks": [{{"name": "...", "present": "...", "absent": "..."}}],
  "final_name": "...", "cause": "...", "category": "...", "confidence": 0-100,
  "severity": "mild | moderate | severe", "reasoning": "signs that confirm it"}}"""
    return _vision(prompt, images_b64)


def _web_search(query, n):
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    return DDGS().text(query, max_results=n) or []


def research(plant, disease, category):
    kind = {"pest": "insecticide", "nutrient deficiency": "fertilizer foliar spray",
            "viral": "vector control", "bacterial": "bactericide copper"}.get((category or "").lower(), "fungicide")
    queries = [f"{plant} {disease} {kind} dose per litre spray",
               f"{plant} {disease} management TNAU agritech",
               f"{plant} {disease} organic control"]
    results, seen = [], set()
    for q in queries:
        try:
            for r in _web_search(q, config.SEARCH_RESULTS_PER_QUERY):
                url = r.get("href") or r.get("url")
                if url and url not in seen:
                    seen.add(url)
                    results.append({"title": r.get("title", ""), "url": url, "snippet": (r.get("body") or "")[:450]})
        except Exception as e:
            print(f"[search] {q!r} failed: {e}")
    return results[:9]


REPORT_SYSTEM = """You are "Leaf Doctor", a kind and experienced agricultural extension officer helping small farmers.
Write in simple English a farmer with basic schooling understands. Short sentences, no jargon.
Rules:
- Be detailed and practical. The farmer must be able to act on it today.
- Give medicines by ACTIVE INGREDIENT (e.g. Mancozeb 75% WP), optionally a common example brand.
- Give doses as amount per 1 litre of water AND per 15-litre knapsack sprayer tank.
- Put the source number like [2] after any dose taken from the web results. If no source gives a dose,
  write "follow the dose printed on the product label" - never invent numbers you are unsure about.
- Do not write any URLs (sources are added automatically).
- If confidence is below 60 or the photo has quality problems, clearly say the diagnosis is not certain,
  mention the other possibilities, ask for a clearer close-up photo and to confirm with the local
  agriculture officer / KVK before buying chemicals."""


def _report_messages(obs, features, final, candidates, quality, sources):
    src = "\n".join(f"[{i}] {s['title']}: {s['snippet']}" for i, s in enumerate(sources, 1)) \
        or "(no web results - use well-established agricultural extension knowledge and say doses must be checked on the label)"
    facts = {
        "plant": obs.get("plant_common_name"), "scientific_name": obs.get("plant_scientific_name"),
        "part": obs.get("plant_part"), "lesion_features": features,
        "diagnosis": final.get("final_name"), "cause": final.get("cause"), "category": final.get("category"),
        "confidence": final.get("confidence"), "severity": final.get("severity"),
        "reasoning": final.get("reasoning"),
        "other_possibilities": [c.get("name") for c in candidates if c.get("name") != final.get("final_name")],
        "photo_quality_problems": quality["issues"],
    }
    user = f"""Diagnosis facts (JSON):
{json.dumps(facts, ensure_ascii=False, indent=1)}

Web research results:
{src}

Write the report in Markdown with exactly these sections:
## Plant identified
## Problem found  (name, cause, type, how sure we are, how serious)
## Signs seen on your plant
## Why this happens and how it spreads
## Do this first (today / next 2 days)
## Medicine (chemical treatment)
   A Markdown table: Medicine (active ingredient) | Dose per 1 L | Dose per 15 L tank | How & when to spray | Repeat after
   Then: best time to spray, how many sprays, days to wait before harvest after spraying.
## Organic / low-cost options
## Safety while spraying
## Prevention for next time
## When to call an expert"""
    return [{"role": "system", "content": REPORT_SYSTEM}, {"role": "user", "content": user}]


def _healthy_messages(obs):
    return [{"role": "user", "content": f"""Write in simple English for a farmer.
A photo of a {obs.get('plant_common_name')} {obs.get('plant_part')} looks HEALTHY.
Observations: {json.dumps(obs.get('symptoms', []), ensure_ascii=False)}
Write a friendly Markdown report: ## Plant identified, ## Health status,
##  Care tips to keep it healthy (water, fertilizer, pests to watch for this crop), ## Early warning signs to watch."""}]


def _sources_md(sources):
    if not sources:
        return ""
    lines = "\n".join(f"{i}. [{s['title'] or s['url']}]({s['url']})" for i, s in enumerate(sources, 1))
    return f"\n\n---\n### Sources\n{lines}"


DISCLAIMER = ("\n\n> AI-based guidance from a photo. Always read the product label, wear protection, "
              "and confirm with your local agriculture officer / Krishi Vigyan Kendra (KVK) before large-scale spraying.")


def _to_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


# ------------------ orchestrator
def run_agent(image: Image.Image):
    """Generator yielding (agent_log_markdown, report_markdown, zoom_crops)."""
    log, crops = [], []

    def step(msg, replace=False):
        if replace and log:
            log[-1] = msg
        else:
            log.append(msg)
        return "\n\n".join(log)

    try:
        # 1. quality
        img = prepare(image)
        quality = quality_check(img)
        q = ", ".join(quality["issues"]) or "good"
        yield step(f"**Step 1/7 Photo check:** {img.size[0]}×{img.size[1]} px, quality: {q}"), "", crops

        # 2. observe
        full_b64, sent_size = _b64(img, config.FULL_IMAGE_SIDE)
        yield step("**Step 2/7 Identifying the plant…**"), "", crops
        obs = observe(full_b64)
        if not obs:
            yield step("The model could not read the image. Try again.", True), "", crops
            return
        if obs.get("is_plant") is False:
            yield step("No plant found in the photo.", True), \
                "### No plant detected\nPlease upload a clear close-up photo of the infected leaf, fruit or plant.", crops
            return
        plant = obs.get("plant_common_name") or "Unknown plant"
        crop_key, entries = kb.lookup(plant, obs.get("plant_scientific_name", ""))
        yield step(f"**Step 2/7 Plant:** {plant} ({obs.get('plant_part')}), "
                   f"confidence {obs.get('plant_id_confidence')}%"
                   + (f" · knowledge base: *{crop_key}*" if crop_key else " · using general knowledge base"), True), "", crops

        # 3. zoom
        yield step("**Step 3/7 Zooming into the affected areas…**"), "", crops
        boxes = _parse_boxes(obs.get("affected_regions"), sent_size)
        source = "AI-selected areas"
        if not boxes:
            boxes, source = _auto_regions(img), "colour analysis"
        crops = [_crop(img, b) for b in boxes]
        crop_b64 = [_b64(c, config.CROP_IMAGE_SIDE)[0] for c in crops]
        images = [full_b64] + crop_b64
        yield step(f"**Step 3/7 {len(crops)} close-ups** from the full-resolution photo ({source})", True), "", crops

        # 4. lesion details
        yield step("**Step 4/7 Studying the spots in detail…**"), "", crops
        features = detail(images, obs)
        short = ", ".join(str(features.get(k)) for k in ("spot_colour", "spot_shape", "spot_size") if features.get(k))
        yield step(f"**Step 4/7 Lesions:** {short or 'described'}", True), "", crops

        # 5. differential diagnosis
        yield step("**Step 5/7 Comparing with known diseases…**"), "", crops
        candidates = diagnose(images, obs, features, entries)
        cand_txt = ", ".join(f"{c.get('name')} ({c.get('confidence')}%)" for c in candidates) or "none"
        yield step(f"**Step 5/7 Possible problems:** {cand_txt}", True), "", crops

        healthy = obs.get("looks_healthy") and (
            not candidates or str(candidates[0].get("name", "")).lower().startswith("healthy"))
        if healthy:
            yield step("**Writing care report…**"), "", crops
            text = ""
            for text in chat_stream(config.REPORT_MODEL, _healthy_messages(obs)):
                yield "\n\n".join(log), text, crops
            yield step("**Done plant looks healthy.**", True), text + DISCLAIMER, crops
            return

        # 6. verify
        yield step("**Step 6/7 Double-checking each sign…**"), "", crops
        final = verify(images, obs, candidates, entries) or {}
        if not final.get("final_name") and candidates:
            top = candidates[0]
            final = {"final_name": top.get("name"), "cause": top.get("cause"), "category": top.get("category"),
                     "confidence": top.get("confidence"), "severity": "unknown", "reasoning": top.get("matching_signs")}
        conf = _to_int(final.get("confidence"))
        if candidates and final.get("final_name") != candidates[0].get("name"):
            conf = min(conf, 65)          # first and second look disagree -> less sure
        if quality["issues"]:
            conf = min(conf, 60)          # poor photo -> less sure
        final["confidence"] = conf
        yield step(f" **Step 6/7 Diagnosis:** {final.get('final_name')} ({final.get('category')}, "
                   f"confidence {conf}%, severity {final.get('severity')})", True), "", crops

        # 7. research + report
        yield step(" **Step 7/7  Searching treatment & writing report…**"), "", crops
        sources = research(plant, final.get("final_name", ""), final.get("category", ""))
        text = ""
        for text in chat_stream(config.REPORT_MODEL,
                                _report_messages(obs, features, final, candidates, quality, sources)):
            yield "\n\n".join(log), text, crops
        yield step(f" **Step 7/7 Report ready** ({len(sources)} web sources)", True), \
            text + _sources_md(sources) + DISCLAIMER, crops

    except OllamaError as e:
        yield step(f"{e}"), "", crops
    except Exception as e:
        yield step(f"Unexpected error: {e}"), "", crops


def check_ollama():
    """Returns a warning string, or '' if everything is ready."""
    try:
        tags = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5).json()
    except Exception:
        return f"Ollama is not reachable at {config.OLLAMA_URL}. Start the app."
    have = {m["name"] for m in tags.get("models", [])}
    missing = [m for m in {config.VISION_MODEL, config.REPORT_MODEL}
               if m not in have and f"{m}:latest" not in have]
    return f"Model not downloaded: {', '.join(missing)}" if missing else ""
