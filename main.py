import cv2
import os
import glob
from config import config
from utils import natural_sort_key, write_log, get_plate_labels, create_output_dirs
from preprocessing import preprocess_plate, remove_small_components
from detector import detect_license_plate
from ocr import get_plate_text, get_actual_plate_text

# configs
IMAGE_ROOT = config["IMAGE_ROOT"]
PLATE_LABELS_PATH = config["PLATE_LABELS_PATH"]
OUTPUT_PATH = config["OUTPUT_PATH"]

def main():
    create_output_dirs()
    
    # get all images and sort an numerical order
    images = sorted(glob.glob(os.path.join(IMAGE_ROOT, "*.jpg")), key=natural_sort_key)
    
    # get plate labels from txt file
    labels = get_plate_labels()

    # assign accuracy calculation variables
    total = correct = incorrect = 0

    for idx, img_path in enumerate(images):
        total += 1
        true_label = labels[idx]

        img = cv2.imread(img_path)

        # get detected plate
        plate_img, bbox = detect_license_plate(img)

        # check an plate image is detected or not
        if plate_img is None:
            print(f"\nIMAGE {os.path.basename(img_path)} → ❌ No plate")
            incorrect += 1
            continue

        # get preprocessed thresholded image
        thresholded = preprocess_plate(plate_img)

        # remove small components (e.g seller brand name) from the plate
        thresholded = remove_small_components(thresholded)

        # get raw plate text
        raw_plate_text = get_plate_text(thresholded)

        # get cleaned plate text (e.g TR at the beginning of the plate was removed)
        cleaned_plate_text = get_actual_plate_text(raw_plate_text)

        # write images
        cv2.imwrite(f"{OUTPUT_PATH}/plates/{idx+1}.jpg", plate_img)
        cv2.imwrite(f"{OUTPUT_PATH}/thresholded/{idx+1}.jpg", thresholded)

        # write result logs into the log file
        write_log(f"IMAGE {os.path.basename(img_path)}")
        write_log(f"  RAW TEXT      : {raw_plate_text}")
        write_log(f"  CLEANED TEXT  : {cleaned_plate_text}")
        write_log(f"  TRUE LABEL   : {true_label}")

        # check plate is identified correctly or not
        if cleaned_plate_text == true_label:
            write_log("  ✔ CORRECT\n")
            correct += 1
        else:
            write_log("  ✖ INCORRECT\n")
            incorrect += 1
            cv2.imwrite(f"{OUTPUT_PATH}/incorrect_thresholded/{idx+1}.jpg", thresholded)
            cv2.imwrite(f"{OUTPUT_PATH}/incorrect_rgb/{idx+1}.jpg", plate_img)
   
   # write final results into the text file
    write_log("\n=============================")
    write_log("       FINAL RESULTS")
    write_log("=============================\n")
    write_log(f"Total images: {total}\n")
    write_log(f"Correct:      {correct}\n")
    write_log(f"Incorrect:    {incorrect}\n")
    write_log(f"Accuracy:     {correct / total * 100:.2f}%")
    write_log("=============================")

if __name__ == "__main__":
    main()