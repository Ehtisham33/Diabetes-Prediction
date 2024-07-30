
# Diabetes Prediction Project

## Overview
This project aims to predict whether a person has diabetes or not using key health metrics such as glucose levels and BMI. The project involves data preprocessing, feature selection, model training, evaluation, and prediction using various machine learning algorithms.

## Table of Contents
- Overview
- Dataset
- Features
- Libraries Used
- Data Preprocessing
- Model Training and Evaluation
- Making Predictions
- Results
- Conclusion
- Usage
- Contributing
- License

## Dataset
The dataset contains health information about individuals, including:
- Glucose levels
- BMI
- Gender
- Age
- Hypertension
- Heart disease
- Smoking history
- HbA1c level
- Diabetes status (target variable)

## Features
The primary features used for prediction in this project are:
- Glucose levels
- BMI

## Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn

## Data Preprocessing
1. **Importing Libraries**: Import necessary libraries for data manipulation and machine learning.
2. **Loading Dataset**: Load the dataset into a pandas DataFrame.
3. **Checking for Null Values**: Identify and handle any missing values.
4. **Checking for Duplicate Values**: Identify and remove duplicate entries.
5. **Checking Data Types**: Ensure all data types are correct for analysis.
6. **Generating Statistical Summaries**: Summarize the data using descriptive statistics.

## Model Training and Evaluation
### 1. Splitting Data
Split the data into training and testing sets:
```python
from sklearn.model_selection import train_test_split
X = df_new.drop(['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'HbA1c_level', 'diabetes'], axis=1)
y = df_new['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. AdaBoostClassifier
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
abc = AdaBoostClassifier()
abc.fit(X_train, y_train)
abc_pred = abc.predict(X_test)
abc_accuracy = accuracy_score(y_test, abc_pred)
print(f"AdaBoostClassifier Accuracy: {abc_accuracy}")
```

### 3. GradientBoostingClassifier
```python
from sklearn.ensemble import GradientBoostingClassifier
gc = GradientBoostingClassifier()
gc.fit(X_train, y_train)
gc_pred = gc.predict(X_test)
gc_accuracy = accuracy_score(y_test, gc_pred)
print(f"GradientBoostingClassifier Accuracy: {gc_accuracy}")
```

### 4. RandomForestClassifier
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"RandomForestClassifier Accuracy: {rf_accuracy}")
```

## Making Predictions
Use the trained GradientBoostingClassifier to make predictions on new data:
```python
import numpy as np
input_data = np.array([[25.19, 140]])
prediction = gc.predict(input_data)
print(f"Prediction: {prediction}")
```

## Results
- **AdaBoostClassifier Accuracy**: 94.57%
- **GradientBoostingClassifier Accuracy**: 94.57%
- **RandomForestClassifier Accuracy**: 92.59%

## Conclusion
The GradientBoostingClassifier and AdaBoostClassifier both achieved the highest accuracy of 94.57%. This project demonstrates the effectiveness of machine learning in predicting diabetes using health metrics.

## Usage
1. Clone the repository:
```bash
git clone https://github.com/Ehtisham33/Diabetes-Prediction.git
```
2. Navigate to the project directory:
```bash
cd Diabetes-Prediction
```
3. Install the required libraries:
```bash
pip install -r requirements.txt
```
4. Run the project:
```bash
python main.py
```

## Contributing
Contributions are welcome! Please create a pull request or open an issue to discuss any changes.

## License
This project is licensed under the MIT License.
