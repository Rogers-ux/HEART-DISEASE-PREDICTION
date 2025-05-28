🫀 Heart Disease Prediction using Machine Learning

This project presents a predictive analysis of heart disease using various machine learning algorithms. The goal is to build an effective classification model that can predict the presence of heart disease in patients based on clinical features.
📊 Dataset

The dataset used is a publicly available heart disease dataset from Kaggle. It contains 14 features including age, sex, chest pain type, blood pressure, cholesterol, fasting blood sugar, ECG results, and more.
🧠 Models Used

The following models were trained and evaluated:

    Support Vector Machine (SVM)

    Random Forest Classifier

    Logistic Regression

    Decision Tree Classifier

    K-Nearest Neighbors (KNN)

Hyperparameter tuning was performed using:

    GridSearchCV for optimal parameter selection

✅ Workflow

    Data Cleaning & Preprocessing

        Handling missing values

        Encoding categorical variables

        Feature scaling

    Exploratory Data Analysis (EDA)

        Correlation heatmaps

        Distribution plots

    Model Training & Evaluation

        Training multiple models

        Evaluating with metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

        Comparing model performance

    Best Model Selection

        GridSearchCV used to fine-tune parameters for the best-performing model

📈 Results

The best performance was observed with the Random Forest Classifier, providing high accuracy and a balanced trade-off between precision and recall. The final model was evaluated using ROC curves for interpretability.
🚀 Getting Started

To run the notebook:

    Clone this repository

    Install required packages using pip install -r requirements.txt

    Open the notebook file and run the cells sequentially

📂 Folder Structure

heart-disease-prediction/
│
├── heart_disease_prediction.ipynb   # Main Jupyter Notebook
├── requirements.txt                 # Dependencies
├── README.md                        # Project overview
└── dataset/                         # Input dataset (optional .csv)

💡 Future Work

    Deploy the best model using Streamlit or Flask

    Implement feature importance visualizations

    Expand the dataset for better generalizability

👨‍💻 Author

This project was created to demonstrate proficiency in applied machine learning and data science. Feel free to fork and contribute.
