# scripts/build_msasl_manifest.py
import json, csv, re, yaml
from pathlib import Path

CFG = yaml.safe_load(open("configs/config.yaml"))
JSON_DIR = Path(CFG.get("microsoft_asl_json_dir", "./data/microsoft_asl/ms_asl_json"))
OUT_CSV  = Path("artifacts/manifests/msasl_all.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def norm(s):
    return s.strip().lower().replace(" ", "_") if isinstance(s, str) else s

def yt_id(url):
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_\-]{6,})", url or "")
    return m.group(1) if m else ""

def load(p): 
    with open(p, "r", encoding="utf-8") as f: 
        return json.load(f)

# --- class id -> gloss ---
raw_classes = load(JSON_DIR/"MSASL_classes.json")
if isinstance(raw_classes, list):
    id2gloss = {i: norm(g) for i, g in enumerate(raw_classes)}
else:
    id2gloss = {int(k): norm(v) for k,v in raw_classes.items()}

# --- synonyms -> canonical ---
syn_map = {}
raw_syn = load(JSON_DIR/"MSASL_synonym.json")
if isinstance(raw_syn, list):
    for row in raw_syn:
        if isinstance(row, list) and row:
            canon = norm(row[0])
            for w in row:
                syn_map[norm(w)] = canon
        elif isinstance(row, dict) and "gloss" in row:
            canon = norm(row["gloss"])
            for w in row.get("synonyms", []):
                syn_map[norm(w)] = canon

# keep only non-letter words from your config labels
allowed = [l for l in CFG["labels"] if not (len(l)==1 and l.isalpha())]
ALLOW = set(norm(x) for x in allowed)

def unify_gloss(item):
    g = item.get("clean_text") or item.get("text")
    g = norm(g) if g else id2gloss.get(int(item.get("label", -1)))
    return syn_map.get(g, g)

def rows_from(split, items):
    for i, x in enumerate(items):
        gloss = unify_gloss(x)
        if ALLOW and gloss not in ALLOW: 
            continue
        url = x.get("url", "")
        if not url: 
            continue
        start  = int(x.get("start", 0))
        end    = int(x.get("end", 0))
        dur    = float(x.get("end_time", 0.0)) - float(x.get("start_time", 0.0))
        if end <= start or dur < 0.5:
            continue
        box = x.get("box") or [None]*4
        yield {
            "sample_id": f"{split}_{i}",
            "split": split,
            "label_text": gloss,
            "label_id": int(x.get("label", -1)),
            "signer_id": int(x.get("signer_id", -1)),
            "fps": float(x.get("fps", 0.0)),
            "width": int(float(x.get("width", 0.0))),
            "height": int(float(x.get("height", 0.0))),
            "start": start, "end": end, "frames": max(0, end-start),
            "start_time": float(x.get("start_time", 0.0)),
            "end_time": float(x.get("end_time", 0.0)),
            "duration_sec": max(0.0, dur),
            "yt_url": url, "yt_id": yt_id(url),
            "box_x": box[0], "box_y": box[1], "box_w": box[2], "box_h": box[3],
            "local_path": ""  # will be filled after download/trim
        }

pieces = []
for split, fname in [("train","MSASL_train.json"),("val","MSASL_val.json"),("test","MSASL_test.json")]:
    f = JSON_DIR/fname
    if f.exists():
        pieces.extend(rows_from(split, load(f)))

cols = ["sample_id","split","label_text","label_id","signer_id","fps","width","height",
        "start","end","frames","start_time","end_time","duration_sec","yt_url","yt_id",
        "box_x","box_y","box_w","box_h","local_path"]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=cols); w.writeheader()
    for r in pieces: w.writerow(r)

print(f"Wrote {len(pieces)} rows → {OUT_CSV}")
