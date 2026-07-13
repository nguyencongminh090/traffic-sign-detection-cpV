import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns

os.makedirs('figures', exist_ok=True)

TRAIN_DIR = "./data/GTSRB/Final_Training/Images"
TEST_DIR = "./data/GTSRB/Final_Test/Images"

IMG_SIZE = (32, 32)

csv_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*", "GT-*.csv")))

print("Generating HOG visualization...")
try:
    sample_csv = csv_files[0]
    df_sample = pd.read_csv(sample_csv, sep=";")
    sample_row = df_sample.iloc[0]
    sample_folder = os.path.dirname(sample_csv)
    sample_image_path = os.path.join(sample_folder, sample_row["Filename"])

    img = cv2.imread(sample_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    roi = (int(sample_row["Roi.X1"]), int(sample_row["Roi.Y1"]), int(sample_row["Roi.X2"]), int(sample_row["Roi.Y2"]))
    x1, y1, x2, y2 = roi
    crop = img[y1:y2+1, x1:x2+1]
    crop_resized = cv2.resize(crop, IMG_SIZE)
    gray = cv2.cvtColor(crop_resized, cv2.COLOR_RGB2GRAY)

    feature, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", visualize=True)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(crop_resized)
    plt.title("Cropped + Resized")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(hog_image, cmap='gray')
    plt.title("HOG Features")
    plt.axis("off")

    plt.savefig('figures/hog_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Success: figures/hog_visualization.png")
except Exception as e:
    print(f"Error generating hog_visualization.png: {e}")

print("Loading model and generating test predictions...")
try:
    model = joblib.load("./hog_svm_gtsrb_model.pkl")
    test_gt_candidates = glob.glob("./data/**/GT-final_test.csv", recursive=True)
    TEST_GT_PATH = test_gt_candidates[0]
    test_df = pd.read_csv(TEST_GT_PATH, sep=";")

    class_names = {i: f"Class {i}" for i in range(43)} # Simplified

    sample_df = test_df.sample(n=12, random_state=42)

    plt.figure(figsize=(18, 12))
    for i, (_, row) in enumerate(sample_df.iterrows()):
        image_path = os.path.join(TEST_DIR, row["Filename"])
        roi = (int(row["Roi.X1"]), int(row["Roi.Y1"]), int(row["Roi.X2"]), int(row["Roi.Y2"]))
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop = img[roi[1]:roi[3]+1, roi[0]:roi[2]+1]
        crop_resized = cv2.resize(crop, IMG_SIZE)
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_RGB2GRAY)
        
        feature = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True).astype("float32")
        
        pred = model.predict([feature])[0]
        true_label = int(row["ClassId"])
        
        result = "Correct" if pred == true_label else "Wrong"
        
        plt.subplot(3, 4, i + 1)
        plt.imshow(crop_resized)
        plt.axis("off")
        plt.title(f"True: {true_label}\nPred: {pred}\n{result}", fontsize=12)

    plt.tight_layout()
    plt.savefig('figures/detection_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Success: figures/detection_examples.png")

    print("Generating Error Analysis (Confusion Matrix)...")
    npz_file = "./gtsrb_hog_32x32_train_features.npz"
    data = np.load(npz_file)
    X, y = data["X"], data["y"]
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_val_pred = model.predict(X_val)
    
    cm = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, cmap='Blues', cbar=False)
    plt.title("Confusion Matrix on Validation Set", fontsize=18)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.savefig('figures/fig_08_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Success: figures/fig_08_error_analysis.png")

except Exception as e:
    print(f"Error generating detection/error plots: {e}")
