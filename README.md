# NewsGuardAI

## Overview
NewsGuardAI is a machine learning-based system designed to classify news articles as **Fake** or **Real**. With the rise of misinformation, this project aims to enhance the detection of deceptive news by leveraging advanced Natural Language Processing (NLP) and machine learning techniques. The goal is to improve classification accuracy compared to existing research.

## Project Pipeline
The project follows a structured approach:

1. **Data Preparation and Cleaning**  
   - Handling missing values, removing duplicates, and preprocessing text.

2. **Exploratory Data Analysis (EDA) & Feature Engineering**  
   - Analyzing data distribution and extracting key features.

3. **Feature Extraction**  
   - Converting text into numerical representations using TF-IDF.

4. **Modeling (Machine Learning-based Classification)**  
   - Training and evaluating models like Logistic Regression, Naive Bayes, and SVM.

5. **Deployment using Streamlit**  
   - A simple web application for real-time news classification.

## Dataset
The dataset consists of labeled news articles with their **title**, **content**, and respective **truthfulness labels** (Fake or Real). Preprocessing techniques ensure optimal feature extraction for improved performance.

## Technologies Used
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-Learn, NLTK, SciPy  
- **Modeling Techniques:** Logistic Regression, Naive Bayes, SVM  
- **Deployment Framework:** Streamlit  

## How to Run

### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/NewsGuardAI.git
cd NewsGuardAI
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```sh
streamlit run Streamlit_Deployment.py
```

## Performance Metrics
The model is evaluated using:
- **F1-score** (Primary metric)
- **Accuracy**
- **Confusion Matrix**

## Contributors
This is a team project for the semester.

## Future Enhancements
- Integrate deep learning models for improved accuracy.
- Expand dataset sources for better generalization.
- Optimize feature engineering techniques.

---
Feel free to contribute and improve NewsGuardAI! 🚀

