# Arrhythmia Detection Using RNNs  
  
This project focuses on detecting and classifying cardiac arrhythmias using Electrocardiogram (ECG) data. A **Recurrent Neural Network (RNN)** with **Bidirectional LSTM** is used to analyze ECG signals and classify them into different arrhythmia types.  

# Technologies Used  
- Python  
- TensorFlow/Keras  
- Scikit-learn  
- NumPy & Pandas  
- Seaborn & Matplotlib  
- SMOTE (for handling class imbalance)  

# Dataset  
The project uses the MIT-BIH Arrhythmia Database, which contains ECG recordings labeled with different arrhythmia types.  

# Installation & Setup  
1️⃣ Clone this repository:  
```bash
git clone https://github.com/your-username/arrhythmia-detection.git
cd arrhythmia-detection
```
2️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
```
3️⃣ Run the prediction script:  
```bash
python prediction.py
```

# Model Training & Evaluation
- The model is trained on ECG data with **categorical cross-entropy loss** and **Adam optimizer**.  
- Evaluation includes accuracy, confusion matrix, and classification report.  

# Results & Visualization
The project includes visualization of:  
- **Training history** (accuracy/loss over epochs)  
- **Confusion matrix**  
- **Classification report**



![Figure_1](https://github.com/user-attachments/assets/05bd4704-eed9-4432-b052-8a9802ae34ea)

