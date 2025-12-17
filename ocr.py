import cv2
import numpy as np
import re
import easyocr
from preprocessing import rotate_img, apply_foreground_crop
from utils import apply_map, swap_character, get_is_binary, clean_text, join_sorted_texts, get_device, convert_to_TR_format

device = get_device()

# Load easy ocr
print("[INFO] Loading OCR model...")
reader = easyocr.Reader(["en"], gpu=device, verbose=False)

# define common provinces in order to solve digitwise confusion in ocr 
COMMON_PROVINCES = ["66","06","34","38","27","35","07","16","01","33","52","20","25","41","17","09","61","50","26","31","48","77"]

# assign rank for each character according to their frequency in the dataset
prov_rank = {p: i for i, p in enumerate(COMMON_PROVINCES)}

# define digit and letter confusion mappings and swaps
DIGITLIKE = {"O": "0", "Q": "0", "D": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}

DIGITS_MAP = {**DIGITLIKE, "L": "4"}  # IMPORTANT: L->4 in digit context
LETTERS_MAP = {"0":"O","1":"I","2":"Z","4":"L","5":"S","6":"G","7":"T","8":"B"}  # digit->letter in letter context

LETTER_SWAPS = {
    "C": ["G"], "G": ["C"],
    "H": ["M", "N"], "M": ["H", "N"], "N": ["M", "H"],
    "Y": ["V"], "V": ["Y"],
}

DIGIT_SWAPS = {
    "0": ["8"], "8": ["0"],
    "6": ["0","4"], "4": ["6"],
}

# assign valid provinces (all provinces must be in between [1,81] )
VALID_PROVINCES = {f"{i:02d}" for i in range(1, 82)}

def get_plate_text(thresh):
    if thresh is None:
        return ""

    if thresh.ndim == 3:
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    if thresh.dtype != np.uint8:
        thresh = np.clip(thresh, 0, 255).astype(np.uint8)

    for raw1, conf1 in get_best_ocr(thresh, strong_only=True):
        plate_text = get_actual_plate_text(raw1)
        # check the plate is valid or not
        if is_valid_tr_plate(plate_text):
            return plate_text

    # improve the plate text quality
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    v_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k2, iterations=1)
    v_dilate = cv2.dilate(thresh, k2, iterations=1)
    v_erode = cv2.erode(thresh, k2, iterations=1)
    _, v_strict = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY)
    v_rot_n2 = rotate_img(thresh, -2.0)
    v_rot_p2 = rotate_img(thresh,  2.0)

    """
        get different variations of images in order to get 
        the most recognizable image by the ocr
    """

    candidates_imgs = [thresh, v_close, v_dilate, v_erode, v_strict, v_rot_n2, v_rot_p2, apply_foreground_crop(thresh)]

    # apply foreground crop in order to improve the accuracy
    if get_is_binary(thresh):
        white_ratio = float((thresh > 127).mean())
        canonical = thresh if white_ratio > 0.5 else cv2.bitwise_not(thresh)
        candidates_imgs.append(canonical)
        candidates_imgs.append(apply_foreground_crop(canonical))

    # get unique images for the processing
    uniq_imgs, seen = [], set()
    for img in candidates_imgs:
        key = img.tobytes()
        if key not in seen:
            seen.add(key)
            uniq_imgs.append(img)

    # get improved versions of the images
    candidates = []

    for v in uniq_imgs:
        for raw, conf in get_best_ocr(v, strong_only=False):
            # baseline interpretation
            actual = get_actual_plate_text(raw)
            candidates.append((raw, actual, conf, 0.0))

            # grammar decode (only if it produces a valid plate)
            decoded, dscore = decode_plate_from_raw(raw, conf)
            if decoded:
                candidates.append((raw, decoded, conf, dscore))

            if not is_valid_tr_plate(actual):
                r1 = repair_missing_province(actual) or repair_missing_province(raw)
                if r1:
                    candidates.append((raw, r1, conf, 5.0))
                r2 = repair_reorder(actual) or repair_reorder(raw)
                if r2:
                    candidates.append((raw, r2, conf, 5.0))

    # pick the best (prefer valid + grammar score)
    best_raw, best_actual, _, _ = max(candidates, key=score_plate)

    if is_valid_tr_plate(best_actual):
        return best_actual.strip()

    return (best_raw or "").strip()

# determine the given plate text is valid or not
def is_valid_tr_plate(plate_text):
    cleaned_text = clean_text(plate_text)
    if len(cleaned_text) < 6:
        return False
    if not cleaned_text[:2].isdigit():
        return False
    if cleaned_text[:2] not in VALID_PROVINCES:
        return False
    rest = cleaned_text[2:]
    has_letter = any(ch.isalpha() for ch in rest)
    has_digit = any(ch.isdigit() for ch in rest)
    return has_letter and has_digit

# use ocr in order to determine plate text
def use_ocr(img, **cfg):
    try:
        output = reader.readtext(
            img,
            detail=1,
            allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            paragraph=False,
            **cfg
        )
    except Exception:
        return "", 0.0
    
    sorted_texts = join_sorted_texts(output)
    return sorted_texts

# get best ocr readable parts of the iamge
def get_best_ocr(img, strong_only=False):

    # apply different decoding methods
    base_cfgs = [
        dict(decoder="beamsearch"),
        dict(decoder="greedy"), 
    ]
    
    # get results of different configs
    results = []
    for cfg in base_cfgs:
        raw, conf = use_ocr(img, **cfg)
        results.append((raw, conf))

    # if score is good apply early stopping
    best_raw, best_conf = max(results, key=lambda x: x[1])
    actual0 = get_actual_plate_text(best_raw)
    if strong_only and best_conf >= 0.86 and is_valid_tr_plate(actual0):
        return [(best_raw, best_conf)]

    # if the score is not good apply other configs
    extra_cfgs = [
        dict(decoder="beamsearch", add_margin=0.18, mag_ratio=2.0, canvas_size=3200),
        dict(decoder="greedy",     add_margin=0.18, mag_ratio=2.0, canvas_size=3200),
        dict(decoder="beamsearch", add_margin=0.18, mag_ratio=2.0, canvas_size=3200,
                text_threshold=0.55, low_text=0.30, link_threshold=0.35),
        dict(decoder="beamsearch", add_margin=0.18, mag_ratio=2.0, canvas_size=3200,
                contrast_ths=0.05, adjust_contrast=0.7),
    ]

    for cfg in extra_cfgs:
        raw, conf = use_ocr(img, **cfg)
        if raw:
            results.append((raw, conf))

    # get all variants and pick up the best one
    uniq, seen = [], set()
    for r, c in results:
        k = (r or "").strip()
        if k not in seen:
            seen.add(k)
            uniq.append((r, c))
    return uniq


def repair_missing_province(text):
    # get cleaned text
    cleaned = clean_text(text)

    if len(cleaned) < 3:
        return ""
    
    c2 = list(cleaned)
    
    # check the plate is in valid format or not
    for i in range(min(2, len(c2))):
        if c2[i] in DIGITLIKE:
            c2[i] = DIGITLIKE[c2[i]]
    c = "".join(c2)

    if len(c) >= 2 and c[:2].isdigit() and c[:2] in VALID_PROVINCES:
        return ""

    # get possible candidates and pick up the best one
    if c[0].isdigit() and len(c) >= 2 and c[1].isalpha():
        d = c[0]
        rest = c[1:]
        candidates = []
        for x in "0123456789":
            prov = f"{x}{d}"
            if prov in VALID_PROVINCES:
                candidates.append(prov)
        for y in "0123456789":
            prov = f"{d}{y}"
            if prov in VALID_PROVINCES:
                candidates.append(prov)
        if not candidates:
            return ""
        candidates.sort(key=lambda p: prov_rank.get(p, 9999))
        best_prov = candidates[0]
        fmt = convert_to_TR_format(best_prov + rest)
        return fmt if is_valid_tr_plate(fmt) else ""

    if c[0].isalpha() and len(c) >= 2 and c[1].isdigit() and c[0] in DIGITLIKE:
        cand = DIGITLIKE[c[0]] + c[1:]
        if len(cand) >= 2 and cand[:2] in VALID_PROVINCES:
            fmt = convert_to_TR_format(cand)
            return fmt if is_valid_tr_plate(fmt) else ""

    return ""
    

# repair plate order if it is wrong
def repair_reorder(text):
    c = clean_text(text)
    m = re.match(r"^(\d{2,4})(\d{2})([A-Z]{1,3})(\d{0,2})$", c)
    if not m:
        return ""
    head, prov, letters, tail = m.groups()
    if prov not in VALID_PROVINCES:
        return ""
    digits = head + tail
    if not (2 <= len(digits) <= 4):
        return ""
    fmt = f"{prov} {letters} {digits}"
    return fmt if is_valid_tr_plate(fmt) else ""

def decode_plate_from_raw(raw, conf):
    cleaned = clean_text(raw)
    if not cleaned:
        return "", -1e18

    # create a set of candidate strings
    trims = {cleaned}

    # check cleaned text is long enough
    if len(cleaned) >= 7:
        # drop first character
        trims.add(cleaned[1:])

        # drop last character
        trims.add(cleaned[:-1])

    # track the best plate and its scoder
    best_plate, best_score = "", -1e18

    # try all valid letter and digit length combinations
    for c in trims:
        if not (5 <= len(c) <= 9):
            continue

        # enumerate splits
        for l_len in (1, 2, 3):
            for d_len in (2, 3, 4):
                
                #skip if total length does not match
                if 2 + l_len + d_len != len(c):
                    continue

                prov_raw = c[:2]
                let_raw  = c[2:2+l_len]
                dig_raw  = c[2+l_len:]

                prov_mapped, prov_cost = apply_map(prov_raw, DIGITLIKE)

                # check province is numeric and valid
                if not (prov_mapped.isdigit() and prov_mapped in VALID_PROVINCES):
                    # allow 1-swap in province digits only if close
                    for pv in swap_character(prov_mapped, DIGIT_SWAPS):
                        if pv.isdigit() and pv in VALID_PROVINCES:
                            prov_mapped = pv
                            prov_cost += 1
                            break
                    else:
                        continue

                # map digits to letters if needed
                let_mapped, let_cost = apply_map(let_raw, LETTERS_MAP)
                if not let_mapped.isalpha():
                    continue

                # map letters to digits
                dig_mapped, dig_cost = apply_map(dig_raw, DIGITS_MAP)
                if not dig_mapped.isdigit():
                    continue

                plate = f"{prov_mapped} {let_mapped} {dig_mapped}"
                if not is_valid_tr_plate(plate):
                    continue

                # swap confused letters and digits
                let_vars = swap_character(let_mapped, LETTER_SWAPS)
                dig_vars = swap_character(dig_mapped, DIGIT_SWAPS)

                # try improved variants
                for Lv in let_vars:
                    if not Lv.isalpha():
                        continue
                    for Dv in dig_vars:
                        if not Dv.isdigit():
                            continue
                        pv_plate = f"{prov_mapped} {Lv} {Dv}"
                        if not is_valid_tr_plate(pv_plate):
                            continue

                        sub_pen = 0
                        if Lv != let_mapped: sub_pen += 1
                        if Dv != dig_mapped: sub_pen += 1

                        # scoring
                        edits = prov_cost + let_cost + dig_cost + sub_pen
                        score = 100.0 + 10.0 * conf - 3.5 * edits

                        # slight preference for common provinces
                        score += 0.2 * (50 - prov_rank.get(prov_mapped, 50))

                        if score > best_score:
                            best_score = score
                            best_plate = pv_plate

    # return best plate with its score
    return best_plate, best_score

def score_plate(item):
    # get parameters from item
    raw, actual, conf, bonus = item
    
    # get cleaned text
    cleaned_text = clean_text(actual)
    
    # if text is too short, give a lower score
    if len(cleaned_text) < 5:
        return -1e9

    score = 0.0
    
    # assign a high score if the plate is valid
    if is_valid_tr_plate(actual):
        score += 90.0

    score += 2.0 * len(cleaned_text)
    score += 6.0 * conf
    score += bonus

    if 7 <= len(cleaned_text) <= 9:
        score += 6.0
    if len(cleaned_text) > 11:
        score -= 8.0
    return score

def get_actual_plate_text(raw_text):
    if not raw_text:
        return ""

    # apply regular expression to the text
    txt = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    n = len(txt)

    best = None
    best_length = -1

    # allow OCR noise before province
    for start in range(0, n):
        rem = txt[start:]
        R = len(rem)

        # try valid letter lengths 1–3
        for L in (3, 2, 1):
            # try valid number lengths 4, 3, 2
            for N in (4, 3, 2):

                if 2 + L + N > R:
                    continue

                prov = rem[:2]
                letters = rem[2:2+L]
                nums = rem[2+L:2+L+N]

                # type checks
                if not prov.isdigit():
                    continue
                if not letters.isalpha():
                    continue
                if not nums.isdigit():
                    continue

                # length of entire plate block
                this_length = 2 + L + N

                # pick longest valid pattern
                if this_length > best_length:
                    best_length = this_length
                    best = (prov, letters, nums)

        # special case: 4-letter noise: ABBJ → ABB
        
        # try shrinking
        if R >= 6:
            prov = rem[:2]
            letters4 = rem[2:6]

            if prov.isdigit() and letters4.isalpha():
                for shrink_L in (3, 2, 1):
                    letters = letters4[:shrink_L]
                    if not letters.isalpha():
                        continue
                    for N in (4, 3, 2):
                        if 2 + shrink_L + N > R:
                            continue
                        nums = rem[2+4:2+4+N]
                        if nums.isdigit():
                            this_length = 2 + shrink_L + N
                            if this_length > best_length:
                                best_length = this_length
                                best = (prov, letters, nums)

    if best is None:
        return raw_text.strip()

    prov, letters, nums = best
    return f"{prov} {letters} {nums}"