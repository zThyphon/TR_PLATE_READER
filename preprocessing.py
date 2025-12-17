import cv2
import numpy as np
from utils import make_odd, get_white_ratio

def rotate_img(img, angle_deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)

# apply foreground crop in order to reduce noise
def apply_foreground_crop(img):
    ink = (img < 128).astype(np.uint8) * 255
    coords = cv2.findNonZero(ink)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    H, W = img.shape[:2]
    if w < 30 or h < 25:
        return img
    if w > 0.98 * W or h > 0.98 * H:
        return img
    cropped = img[y:y + h, x:x + w]
    pad = 6
    bordered = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return bordered

# crop the image in order to get rid of the brands around the plate
def remove_brands_around_plate(img, x1, y1, x2, y2, margin=40):
    h, w = img.shape[:2]

    ex1 = max(0, x1 - margin)
    ey1 = max(0, y1 - margin)
    ex2 = min(w, x2 + margin)
    ey2 = min(h, y2 + margin)

    cleaned = img.copy()
    cleaned[ey1:ey2, ex1:ex2] = 0
    cleaned[y1:y2, x1:x2] = img[y1:y2, x1:x2]

    return cleaned

def sauvola_threshold(img, window=25, k=0.2):
    # convert to float
    img = img.astype(np.float32)

    # local mean
    mean = cv2.boxFilter(img, ddepth=-1, ksize=(window, window))

    # local variance → std
    mean_sq = cv2.boxFilter(img * img, ddepth=-1, ksize=(window, window))
    var = mean_sq - mean * mean
    var[var < 0] = 0  # numeric safety
    std = np.sqrt(var)

    # Sauvola formula
    R = 128  # dynamic range
    thresh = mean * (1 + k * ((std / R) - 1))

    # produce binary image (invert because characters are dark)
    binary = (img < thresh).astype(np.uint8) * 255

    return binary

# get how much of the foreground components are tiny 
def get_cc_stats(bin_img):
    # create a foreground mask
    fg = (bin_img > 0).astype(np.uint8)

    # find connected components 
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg, connectivity=8
    )   

    # extract component areas
    areas = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.array([], dtype=np.int32)
    
    # count foreground components
    count = int(areas.size)

    Hc, Wc = bin_img.shape[:2]

    # define tiny means
    tiny_thr = max(6, (Hc * Wc) // 2500)

    # compute fraction of tiny components
    tiny_frac = float((areas < tiny_thr).sum()) / max(count, 1)
    return count, tiny_frac

def rescue_edge_strips(binary, gray_src):
    Hh, Ww = binary.shape[:2]
    if Ww < 40 or Hh < 12:
        return binary

    output = binary
    output = rescue_one_side(output, gray_src, side="left")
    output = rescue_one_side(output, gray_src, side="right")
    return output

def rescue_one_side(bin_img, gray_img, side="left"):
        # get height(Hs) and width(Ws) of the binary image
        Hs, Ws = bin_img.shape[:2]

        """
        Compute the width of the edge strip to analyze:
            - At least 4 pixels wide
            - At most 70 pixels wide
            - About 12.5% of the image width
        """

        strip_w = max(4, min(Ws // 8, 70))

        # select which side of the image to process
        if side == "left":
            # left edge strip
            xs = slice(0, strip_w)
        else:
            # right edge strip
            xs = slice(Ws - strip_w, Ws)

        # extract the binary strip from the selected side
        b_strip = bin_img[:, xs]

        # extract the grayscale strip from the same region
        g_strip = gray_img[:, xs]

        # check how much ink we currently have in this strip
        fg_ratio = float((b_strip > 0).sum()) / b_strip.size

        # if there is *plenty* of ink here, we don't need rescue.
        if fg_ratio > 0.08:
            return bin_img

        # check local contrast in grayscale; lower the bar a bit
        g5, g95 = np.percentile(g_strip, (5, 95))
        if (g95 - g5) < 18.0:
            return bin_img
        
        # first apply a small Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(g_strip, (3, 3), 0)
        
        # perform local Otsu thresholding on the strip
        _, strip_bin = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        # convert thresholded strip into a binary mask
        fg = (strip_bin > 0).astype(np.uint8)

        # find connected components using 8-connectivity
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            fg, connectivity=8
        )

        # create an empty image for cleaned components
        cleaned = np.zeros_like(strip_bin)
        if num > 1:
            for lbl in range(1, num):
                x, y, w, h, area = stats[lbl]
                # Slightly relaxed height and area filters
                if h >= int(0.35 * Hs) and area >= max(12, (Hs * strip_w) // 300):
                    cleaned[labels == lbl] = 255

        # ink sanity check
        ink_ratio = float((cleaned > 0).sum()) / cleaned.size
        
        # allow smaller characters (down to ~0.5%) and up to 45% fill
        if ink_ratio < 0.005 or ink_ratio > 0.45:
            return bin_img
        
        merged = bin_img.copy()
        # merge images
        merged[:, xs] = np.maximum(merged[:, xs], cleaned)
        return merged


def preprocess_plate(plate_img):
    # apply size-aware upscale
    h0, w0 = plate_img.shape[:2]
    min0 = min(h0, w0)
    scaled = False

    # for really tiny plates, bring the *smallest* dimension up to ~64
    if min0 < 32:
        target_min = 64
        scale = float(target_min) / max(float(min0), 1.0)
        # don't explode tiny images
        scale = min(scale, 4.0)
        new_w = max(1, int(round(w0 * scale)))
        new_h = max(1, int(round(h0 * scale)))
        plate_img = cv2.resize(
            plate_img, (new_w, new_h),
            interpolation=cv2.INTER_LANCZOS4
        )
        scaled = True
    
    # slight boost for mid-small plates
    elif min0 < 48:
        target_min = 80
        scale = float(target_min) / max(float(min0), 1.0)
        scale = min(scale, 2.5)
        new_w = max(1, int(round(w0 * scale)))
        new_h = max(1, int(round(h0 * scale)))
        plate_img = cv2.resize(
            plate_img, (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )
        scaled = True

    # make image grayscale
    if plate_img.ndim == 2:
        gray = plate_img.copy()
    else:
        if plate_img.shape[2] == 4:
            plate_img = plate_img[:, :, :3]
        # in practice frames are BGR from OpenCV, so use BGR → gray.
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # contrast notmalization
    p2, p98 = np.percentile(gray, (2, 98))
    span = float(p98 - p2)

    if span < 30:
        # low contrast → stronger CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    elif span < 70:
        # medium contrast → softer CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    else:
        # already good contrast → simple linear stretch
        gray = np.clip(
            (gray - p2) * 255.0 / (span + 1e-6),
            0, 255
        ).astype(np.uint8)

    # apply canny edge detection
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))

    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    thickness = float(np.mean(dist))

    # light denoising
    if thickness < 1.6:
        h = 1
    elif thickness < 3.0:
        h = 2
    else:
        h = 3

    denoised = cv2.fastNlMeansDenoising(
        gray, None, h=h,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # apply adaptive thresholding
    H, W = denoised.shape[:2]

    # stroke window size based on "thickness"
    if thickness < 1.6:
        win = 17
        k = 0.22
    elif thickness < 3.0:
        win = 27
        k = 0.16
    else:
        win = 35
        k = 0.12

    # clamp window so it is not insane for tiny plates
    size_limit_h = make_odd(max(15, min(35, H // 2)))
    size_limit_w = make_odd(max(15, min(35, W // 6)))
    size_limit = min(size_limit_h, size_limit_w)
    win = min(win, size_limit)

    # sauvola_threshold is assumed to exist in your environment
    sauvola = sauvola_threshold(denoised, window=win, k=k)

    if sauvola.dtype != np.uint8:
        sauvola = np.clip(sauvola, 0, 255).astype(np.uint8)

    # get white ratio
    wr_s = get_white_ratio(sauvola)

    # slightly stricter bounds than before
    sauvola_looks_bad = (wr_s < 0.03) or (wr_s > 0.55)


    if sauvola_looks_bad:
        blur_fb = cv2.GaussianBlur(denoised, (5, 5), 0)
        _, otsu_fb = cv2.threshold(
            blur_fb, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        wr_o = get_white_ratio(otsu_fb)
        target = 0.18  # typical ink coverage for clean plates
        
        # pick whichever is closer to a reasonable ink coverage
        if abs(wr_o - target) + 0.005 < abs(wr_s - target):
            sauvola = otsu_fb
            wr_s = wr_o

    # optional thin-stroke stitch
    thinish = thickness < 1.8

    # only try this on "normal" plates, not aggressively upscaled tiny ones
    if thinish and (not scaled) and H >= 60 and W >= 140:
        cc_s, tiny_s = get_cc_stats(sauvola)

        k_stitch = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        stitched = cv2.morphologyEx(sauvola, cv2.MORPH_CLOSE, k_stitch, iterations=1)

        wr_t = get_white_ratio(stitched)
        cc_t, tiny_t = get_cc_stats(stitched)

        improved_fragmentation = (cc_t < 0.85 * max(cc_s, 1)) or (tiny_t + 0.03 < tiny_s)
        small_ink_bump = (wr_t - wr_s) <= 0.05

        stitch_ok = (
            0.05 <= wr_s <= 0.22 and
            wr_t <= 0.26 and
            improved_fragmentation and
            small_ink_bump
        )

        if stitch_ok:
            sauvola = stitched
            wr_s = wr_t

    """ 
    for very thin images ipscaled plates, 
    apply a tiny dilation helps fill gaps
    """

    if thickness < 1.6 and scaled:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        sauvola = cv2.dilate(sauvola, kernel, iterations=1)

    # for normal thin plates, micro-dilation only
    elif thickness < 1.6:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        sauvola = cv2.dilate(sauvola, kernel, iterations=1)

    # pptional speckle cleanup, only when edge map suggests heavy noise
    if (thickness >= 1.8 and edge_density > 0.24 and span < 95):
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        sauvola = cv2.morphologyEx(sauvola, cv2.MORPH_OPEN, k_open, iterations=1)

    sauvola = rescue_edge_strips(sauvola, denoised)

    # if inner columns have ink but right edge is almost empty
    # try edge-strip rescue once more
    H2, W2 = sauvola.shape[:2]
    if W2 >= 40:
        col_sum = (sauvola > 0).sum(axis=0).astype(np.float32)
        mid = col_sum[int(W2 * 0.2):int(W2 * 0.8)]
        right = col_sum[int(W2 * 0.8):]

        if mid.size > 0 and right.size > 0:
            mid_mean = float(mid.mean())
            right_mean = float(right.mean())

            if mid_mean > 0 and right_mean < 0.25 * mid_mean:
                sauvola = rescue_edge_strips(sauvola, denoised)

    # return clean binary
    return np.clip(sauvola, 0, 255).astype("uint8")

def remove_small_components(thresh):
    """
        remove obviously tiny noise blobs, but keep real characters.
        this uses *relative* size (median connected-component area)
        instead of hard absolute fractions of the plate height/width
    """

    if thresh.dtype != np.uint8:
        thresh = np.clip(thresh, 0, 255).astype(np.uint8)

    # work on foreground mask (text is white)
    fg = (thresh > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg, connectivity=8
    )

    # nothing or only background (thresholding)
    if num_labels <= 1:
        return thresh

    H, W = thresh.shape[:2]
    cleaned = np.zeros_like(thresh)

    # stats[0] is background
    areas = stats[1:, cv2.CC_STAT_AREA]

    # if all areas are weird, just return original
    med_area = float(np.median(areas))
    if med_area <= 0:
        return thresh

    # smaller than the "typical" component on this plate.
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # ultra-tiny blobs by absolute area
        if area < 4:
            continue

        # very small compared to the typical component => likely noise
        if area < 0.10 * med_area:
            continue

        # keep everything else (characters, joined strokes, etc.)
        cleaned[labels == i] = 255

    return cleaned

def remove_small_components(thresh):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    H, W = thresh.shape
    cleaned = np.zeros_like(thresh)

    # character height range (fraction of plate height)
    MIN_CHAR_H = H * 0.25
    MAX_CHAR_H = H * 0.95

    # character width range (fraction of plate width)
    MIN_CHAR_W = W * 0.02
    MAX_CHAR_W = W * 0.25

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if MIN_CHAR_H <= h <= MAX_CHAR_H and MIN_CHAR_W <= w <= MAX_CHAR_W:
            cleaned[labels == i] = 255

    return cleaned