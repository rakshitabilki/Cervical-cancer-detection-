Cervical Cancer Detection System
AI-powered cervical cancer detection and classification system using Machine Learning and Deep Learning models. This project helps in early diagnosis and screening by analyzing cervical cell images and predicting whether the cell is normal or abnormal.
📌 Features
Upload cervical cell images
Predict normal or abnormal cells
Multiple ML/DL models supported:
Logistic Regression
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Random Forest
Convolutional Neural Network (CNN)
Model comparison dashboard
Accuracy visualization using charts
Interactive Streamlit web application
User-friendly dark themed UI
🧠 Models Used
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 74.42% |
| SVM | 75.19% |
| KNN | 79.84% |
| Random Forest | 86.82% |
| CNN | 74.42% |
🏆 Best Performing Model: Random Forest (86.82%)

🖥️ Tech Stack
- Python
- Streamlit
- Scikit-learn
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib


📂 Project Structure
```
Cervical-Cancer-Detection/
│
├── app.py
├── models/
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   ├── knn.pkl
│   ├── random_forest.pkl
│   └── cnn_model.h5
│
├── dataset/
├── images/
├── requirements.txt
├── README.md
└── notebooks/
```
🚀 Installation
1️⃣ Clone the Repository
```
git clone https://github.com/your-username/cervical-cancer-detection.git
```
2️⃣ Navigate to Project Folder
```
cd cervical-cancer-detection
```
3️⃣ Create Virtual Environment
```
python -m venv .venv
```
4️⃣ Activate Virtual Environment
Windows
```
.venv\Scripts\activate
```
Linux / Mac
```
source .venv/bin/activate
```
📦 Install Dependencies
```
pip install -r requirements.txt
```
▶️ Run the Application
```
streamlit run app.py
```
The application will start at:
```
http://localhost:8501
```
for model training :
```
python train models.py
```
📸 Application Screenshots
Main Interface
Upload cervical cell images
Select ML/DL model
View prediction results
Dashboard
Compare model accuracies
Visualize performance graphs
Identify best model
🔍 Working Process
User uploads cervical cell image
Image preprocessing is performed
Selected model analyzes the image
Prediction is generated
Result displayed as:
Normal Cell Detected
Abnormal Cell Detected
🎯 Use Case
This system can assist in:
Early cervical cancer screening
Medical image classification research
AI-based healthcare applications
Educational and academic projects
📊 Future Improvements
Improve CNN accuracy
Add real-time prediction confidence score
Deploy using Streamlit Cloud or Render
Add patient report generation
Integrate larger medical datasets
⚠️ Disclaimer
This project is developed for educational and research purposes only. It is not intended to replace professional medical diagnosis.
👩‍💻 Author
Developed by Rakshita Bilki
⭐ GitHub
If you like this project, give it a ⭐ on GitHub.
