<<<<<<< HEAD
# FloodNet ML Project — README
## Setup Instructions

### 1. Place this entire folder at:
```
D:\Projects\ML\
```

### 2. Fill Dataset Folders with your data:
```
FloodNet Challenge - Track 1\Train\Labeled\Flooded\image\     → put *.jpg images
FloodNet Challenge - Track 1\Train\Labeled\Flooded\mask\      → put *_lab.png masks
FloodNet Challenge - Track 1\Train\Labeled\Non-Flooded\image\ → put *.jpg images
FloodNet Challenge - Track 1\Train\Labeled\Non-Flooded\mask\  → put *_lab.png masks
FloodNet Challenge - Track 2\Images\Train_Image\              → put *.JPG images
FloodNet Challenge - Track 2\Questions\                       → put Training Question.json
test_images\                                                  → put any test *.jpg
```

### 3. Copy your existing .pkl models to models\ folder:
```
models\task1_model.pkl
models\task2_model.pkl
models\task3_model.pkl
models\task3_tfidf.pkl
models\task3_label_encoder.pkl
```

### 4. Install dependencies:
```bash
pip install scikit-learn scikit-image opencv-python joblib numpy matplotlib flask flask-cors
```

### 5. Start Flask backend:
```bash
cd D:\Projects\ML\app
python app.py
```

### 6. Open browser:
```
http://localhost:5000
```

## Accuracies Achieved
- Task 1 Classification : 96.25%
- Task 2 Segmentation   : 94.07%
- Task 3 VQA            : >75%
=======
# Aerial-FloodNet
>>>>>>> 061f957dd4f33f88ef57b5ea520f5b580d0bb8d6
