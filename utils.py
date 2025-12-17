import os
import re 
import numpy as np
import torch
from config import config

# get log file from config
LOG_FILE = config["LOG_FILE"]

# get plate label txt file path from config
PLATE_LABELS_PATH = config["PLATE_LABELS_PATH"]

# get image output path from config
OUTPUT_PATH = config["OUTPUT_PATH"]

# determine device
def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

# create output directories for writing images into them
def create_output_dirs():
    os.makedirs(f"{OUTPUT_PATH}/plates", exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/thresholded", exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/incorrect_thresholded", exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/incorrect_rgb", exist_ok=True)

# get plate labels from the txt file
def get_plate_labels():
    with open(PLATE_LABELS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]
    
# write log file into the 
def write_log(text):
    with open(LOG_FILE, "a") as file:
        file.write(text)

# apply sorting in a numerical order
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)]

# apply mapping
def apply_map(string, map):
    out = []
    cost = 0

    for ch in string:
        if ch in map:
            out.append(map[ch])
            cost += 1
        else:
            out.append(ch)
    return "".join(out), cost

def swap_character(string, swaps):
    string = string or ""
    output = { string }
    for i, ch in enumerate(string):
        if ch in swaps:
            for r in swaps[ch]:
                output.add(string[:i] + r + string[i+1:])
    return output

def get_is_binary(img):
    return len(np.unique(img)) <= 20

def clean_text(text):
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())

# ensure the number is odd
def make_odd(x):
    return x if x % 2 == 1 else max(1, x - 1)

def get_white_ratio(b):
    return float((b > 0).sum()) / b.size

# merge texts of the image
def join_sorted_texts(img):

    if not img:
        return "", 0.0

    items = []
    for (bbox, txt, conf) in img:
        if not txt:
            continue
        pts = np.array(bbox, dtype=np.float32)
        xs = pts[:, 0]; ys = pts[:, 1]
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        h = max(1.0, y1 - y0)
        cy = 0.5 * (y0 + y1)
        items.append((cy, x0, h, str(txt), float(conf)))

    if not items:
        return "", 0.0

    med_h = float(np.median([it[2] for it in items])) if len(items) else 1.0
    line_tol = 0.60 * max(1.0, med_h)

    items.sort(key=lambda t: t[0])  # cy
    lines = []
    for it in items:
        placed = False
        for L in lines:
            if abs(it[0] - L["cy"]) <= line_tol:
                L["items"].append(it)
                L["cy"] = (L["cy"] * (len(L["items"]) - 1) + it[0]) / len(L["items"])
                placed = True
                break
        if not placed:
            lines.append({"cy": it[0], "items": [it]})

    lines.sort(key=lambda L: L["cy"])
    texts, confs = [], []
    for L in lines:
        L["items"].sort(key=lambda t: t[1])  # x0
        texts.extend([t[3] for t in L["items"]])
        confs.extend([t[4] for t in L["items"]])

    raw = " ".join(texts).strip()
    conf = float(np.mean(confs)) if confs else 0.0
    return raw, conf

# convert the plate into standard TR plate format
def convert_to_TR_format(plate_text):
    prov = plate_text[:2]
    rest = plate_text[2:]
    i = 0
    while i < len(rest) and rest[i].isalpha():
        i += 1
    letters = rest[:i]
    digits = rest[i:]
    if not letters or not digits:
        return ""
    return f"{prov} {letters} {digits}"