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

os.makedirs('figures/enhanced', exist_ok=True)

TRAIN_DIR = "./data/GTSRB/Final_Training/Images"
TEST_DIR = "./data/GTSRB/Final_Test/Images"
IMG_SIZE = (32, 32)

class_names = {
    0: "Speed limit 20 km/h", 1: "Speed limit 30 km/h", 2: "Speed limit 50 km/h",
    3: "Speed limit 60 km/h", 4: "Speed limit 70 km/h", 5: "Speed limit 80 km/h",
    6: "End of speed limit 80 km/h", 7: "Speed limit 100 km/h", 8: "Speed limit 120 km/h",
    9: "No passing", 10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at next intersection", 12: "Priority road", 13: "Yield",
    14: "Stop", 15: "No vehicles", 16: "Vehicles over 3.5 tons prohibited",
    17: "No entry", 18: "General caution", 19: "Dangerous curve to the left",
    20: "Dangerous curve to the right", 21: "Double curve", 22: "Bumpy road",
    23: "Slippery road", 24: "Road narrows on the right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed and passing limits", 33: "Turn right ahead",
    34: "Turn left ahead", 35: "Ahead only", 36: "Go straight or right",
    37: "Go straight or left", 38: "Keep right", 39: "Keep left",
    40: "Roundabout mandatory", 41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 tons"
}

# 1. Enhanced Figure 6: Detection Examples with Bounding Boxes
print("Generating Enhanced Figure 6...")
try:
    model = joblib.load("./hog_svm_gtsrb_model.pkl")
    test_gt_candidates = glob.glob("./data/**/GT-final_test.csv", recursive=True)
    TEST_GT_PATH = test_gt_candidates[0]
    test_df = pd.read_csv(TEST_GT_PATH, sep=";")
    sample_df = test_df.sample(n=6, random_state=42)

    plt.figure(figsize=(15, 10))
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
        
        color = (0, 255, 0) if pred == true_label else (255, 0, 0)
        cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), color, max(1, img.shape[0]//100))
        
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")
        title = f"True: {class_names[true_label]}\nPred: {class_names[pred]}"
        plt.title(title, fontsize=12, color='green' if pred == true_label else 'red')

    plt.tight_layout()
    plt.savefig('figures/enhanced/detection_examples_v2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Success: figures/enhanced/detection_examples_v2.png")
except Exception as e:
    print(f"Error Figure 6: {e}")

# 2. Enhanced Figure 8: Error Analysis (Top Confused Classes)
print("Generating Enhanced Figure 8...")
try:
    npz_file = "./gtsrb_hog_32x32_train_features.npz"
    data = np.load(npz_file)
    X, y = data["X"], data["y"]
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_val_pred = model.predict(X_val)
    
    cm = confusion_matrix(y_val, y_val_pred)
    np.fill_diagonal(cm, 0)
    
    confused_pairs = []
    for i in range(43):
        for j in range(43):
            if cm[i, j] > 0:
                confused_pairs.append((i, j, cm[i, j]))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    top_5 = confused_pairs[:5]
    
    fig = plt.figure(figsize=(16, 12))
    ax1 = plt.subplot(3, 1, 1)
    labels = [f"True: {class_names[t]}\nPred: {class_names[p]}" for t, p, c in top_5]
    counts = [c for t, p, c in top_5]
    
    sns.barplot(x=counts, y=labels, ax=ax1, color="salmon")
    ax1.set_title("Top 5 Confused Class Pairs in Validation Set", fontsize=16)
    ax1.set_xlabel("Number of Misclassifications", fontsize=12)
    
    import textwrap
    for idx, (t, p, c) in enumerate(top_5):
        # True Image
        class_folder_t = os.path.join(TRAIN_DIR, f"{t:05d}")
        img_files_t = glob.glob(os.path.join(class_folder_t, "*.ppm"))
        if img_files_t:
            img_path_t = img_files_t[0]
            img_t = cv2.imread(img_path_t)
            img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
            ax_true = plt.subplot(3, 5, 5 + idx + 1)
            ax_true.imshow(img_t)
            ax_true.axis("off")
            true_title = textwrap.fill(class_names[t], width=15)
            ax_true.set_title(f"True:\n{true_title}", fontsize=10)
            
        # Predicted (Confused) Image
        class_folder_p = os.path.join(TRAIN_DIR, f"{p:05d}")
        img_files_p = glob.glob(os.path.join(class_folder_p, "*.ppm"))
        if img_files_p:
            img_path_p = img_files_p[0]
            img_p = cv2.imread(img_path_p)
            img_p = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)
            ax_pred = plt.subplot(3, 5, 10 + idx + 1)
            ax_pred.imshow(img_p)
            ax_pred.axis("off")
            pred_title = textwrap.fill(class_names[p], width=15)
            ax_pred.set_title(f"Confused As:\n{pred_title}", fontsize=10, color='red')
            
    plt.tight_layout()
    plt.savefig('figures/enhanced/fig_08_error_analysis_v3.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Success: figures/enhanced/fig_08_error_analysis_v3.png")
    
except Exception as e:
    print(f"Error Figure 8: {e}")
