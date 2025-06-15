Fire Detection using AI (CNN Model)
This project demonstrates a real-time fire detection system using Convolutional Neural Networks (CNNs) trained on fire and non-fire images. It classifies whether an image contains fire or not, helping in early fire detection using artificial intelligence.

Features
Detects presence of fire in images using deep learning
Uses a custom dataset of fire and non-fire images
Trained model is saved as .h5 to avoid retraining
Predicts fire from any input image using a simple script
Visualizes training and validation accuracy using graphs

AI/ML Concepts Used
Convolutional Neural Networks (CNN)
Image preprocessing using OpenCV
Model evaluation with accuracy and loss metrics
Model saving and loading using TensorFlow
Inference on new test images

Project Structure
FireDetectionAI/
│
├── fire_dataset/                
├── fire_detection_model.h5      
├── model_training.py               
├── fire_predict.py               
├── fire1.jpg,fire2.jpg,fire3.jpg
└── README.md                    
How to Use
1. Clone the Repository
git clone https://github.com/jayassurya/Fire_Detection_System.git
cd Fire_Detection_System

2. Install Dependencies
pip install tensorflow opencv-python numpy matplotlib

3. Train the Model (Optional)
python train_model.py

4. Run Fire Detection
python detect_fire.py
Note: Place your test image (e.g., fire1.jpg) inside the test_images/ folder and update the image path in the script if necessary.

Sample Output
Test Accuracy: 90.25%
Prediction: Fire detected in fire1.jpg

Performance Comparison
Metric	Value
Accuracy	90.25%
Precision	88.50%
Recall	91.30%
F1 Score	89.85%
