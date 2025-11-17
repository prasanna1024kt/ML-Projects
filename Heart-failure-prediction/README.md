# Heart Failure Prediction

## Context:
This dataset contains the medical records of 5000 patients who had heart failure, collected during their follow-up period, where each patient profile has 13 clinical features.

## Problem Statement

Heart failure is a life-threatening condition that requires early detection to improve patient outcomes. However, identifying high-risk patients from clinical data is challenging due to multiple overlapping medical factors.  

This project aims to develop a machine learning model that predicts the risk of heart failure using patient clinical records and exposes a simple API for real-time predictions, enabling faster and more informed medical decision-making.

## Project Components 

### **1. Data Pipeline**
- Loads raw heart failure dataset.
- Cleans and preprocesses data (missing value handling, encoding).
- Splits data into training and testing sets.

### **2. Data Model**
- Trains ML models (Logistic Regression, Random Forest, Decision Tree, XGBoost).
- Uses DictVectorizer for feature transformation.
- Evaluates models (Accuracy, ROC-AUC).
- Saves trained model artifacts as `.bin` files.

### **3. Application Backend**
- Flask API that loads the trained model and serves predictions.
- Provides endpoints for:
  - `/predict` – predict heart failure risk
  - `/metrics` – return model accuracy/ROC
  - `/features` – list expected input features

### **4. Containerisation**
- Docker image packages the Flask API and model artifacts.
- Ensures consistent deployment.
- Run using:  
  ```  docker run -p 9696:9696 heart-medical-records ```

## Project Structure

The project is organized as follows:

### **Notebooks**
- **heart_failure-EDA.ipynb**  
  Contains exploratory data analysis (EDA), feature engineering.
### **Scripts**
Located inside the `model/` folder:
- **train.py**  
  Handles:
  - Data loading  
  - Preprocessing  
  - Train/test split  
  - Model training  
  - Model evaluation  
  - Saving model artifacts (`.bin` files)
- **predict.py**  
  Flask-based inference API that:
  - Loads the trained `.bin` model artifact  
  - Applies preprocessing  
  - Exposes prediction, metrics, and feature endpoints  

### **Configuration Files**
- **Pipfile**  
  Defines project dependencies and Python version (managed by Pipenv).

- **Pipfile.lock**  
  Locks exact dependency versions for deterministic and repeatable builds.

- **requirements.txt** *(optional)*  
  Used mainly for Docker builds if necessary.

- **Dockerfile**  
  Defines steps to containerize the Flask prediction API.

### **Dataset**
- **heart_failure_clinical_records.csv**  
  Located under:Datasets folder
---

## About Dataset

### Attribute Information:
| Feature Name              | Description |
|---------------------------|-------------|
| age                       | Age of the patient (years) |
| anaemia                   | Decrease of red blood cells or hemoglobin (boolean) |
| creatinine_phosphokinase | Level of the CPK enzyme in the blood (mcg/L) |
| diabetes                  | Whether the patient has diabetes (boolean) |
| ejection_fraction         | Percentage of blood leaving the heart at each contraction (%) |
| high_blood_pressure       | Whether the patient has hypertension (boolean) |
| platelets                 | Platelets in the blood (kiloplatelets/mL) |
| sex                       | Sex of the patient (0 = female, 1 = male) |
| serum_creatinine          | Level of serum creatinine in the blood (mg/dL) |
| serum_sodium              | Level of serum sodium in the blood (mEq/L) |
| smoking                   | Whether the patient smokes (boolean) |
| time                      | Follow-up period (days) |
| DEATH_EVENT               | If the patient died during follow-up (0 = survived, 1 = died) |

## Data Analysis (EDA)

### Correlation 
![Model Plot](images/Correlation.png)

### Pairplot of heart failure 
![Model Plot](images/pair_plot_heart_failure_top.png) 

## Key Insights from the Dataset

The following clinical factors show strong relationships with heart failure outcomes:

| Factor | Observation | Interpretation |
|--------|-------------|----------------|
| **Older Age** | Higher mortality | Aging hearts are less resilient and prone to failure. |
| **Low Ejection Fraction** | Strong predictor of death | Direct measure of reduced heart pumping strength. |
| **High Serum Creatinine** | Higher mortality | Indicates impaired kidney function, often linked with heart failure severity. |
| **Low Serum Sodium** | Higher mortality | Sign of advanced heart failure and fluid imbalance. |
| **Follow-up Time** | Inversely correlated with survival | Severe cases tend to result in earlier mortality. |
| **Lifestyle Factors (Smoking, Sex, Diabetes)** | Weak correlation | Limited predictive power compared to core clinical metrics. |

### Key prediction of death 
    The model identifies the following clinical features as the strongest indicators of heart failure–related mortality:

- **Low ejection fraction** – indicates poor heart pumping capacity  
- **High serum creatinine** – reflects kidney dysfunction  
- **Older age** – increased vulnerability to cardiac events  
- **Low serum sodium** – associated with severe heart failure  
- **Shorter follow-up time** – severe cases deteriorate earlier  

### Model comparision 
    | Model Name           | Accuracy | ROC       |
    |----------------------|----------|-----------|
    | Logistic Regression  | 0.833    | 0.895     |
    | Random Forest        | 0.982    | 0.98979   |
    | Decision Tree        | 0.987    | 0.98503   |
    | XGBoost              | 0.992    | 0.999425  |

### Best Model Selection 

``` xgboost is the best model with Accuracy 0.992 and roc 0.999425 ```

---

## Instructions to Run the Application

### Prerequisites
- Python 3.12.3: Ensure Python is installed on your machine.
- Pipenv: Install Pipenv for dependency management.
- Docker: To run the application in a containerized environment.

### Local Setup 

#### 1. Clone repo 

```markdown
git clone https://github.com/prasanna1024kt/ML-Projects.git
cd ML-Projects/Heart-failure-prediction
```
#### 2. Install dependencies
```markdown
pip install pipenv
pipenv install --system --develop
```
#### 3. Run the application

``` gunicorn --bind=0.0.0.0:9696 predict:app ```

#### 4. Test locally 
```
http://localhost:9696/metrics 
http://localhost:9696/predicts 
http://localhost:9696/features 
```

### Using Docker 

1. Build the Docker container:

``` docker build -t heart-medical-records . ```

2. Run the container:

``` docker run -p 9696:9696 heart-medical-records ```


### Test Deployment 

1. Using cURL Send a POST request to test the API:

```  curl -s -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"data": 
  {"age":60,
  "anaemia":0,
  "creatinine_phosphokinase":250,
  "diabetes":0,
  "ejection_fraction":38,
  "high_blood_pressure":0,
  "platelets":260000,
  "sex":1,
  "serum_creatinine":1.1,
  "serum_sodium":137,
  "smoking":0,
  "time":45}
  }' | jq
```
OutPut: 

``` 
{
  "model_metrics": {
    "accuracy": 0.992,
    "roc_auc": 0.9994259853128142,
    "threshold": 0.5
  },
  "results": [
    {
      "class_label": 1,
      "heart_failure": true,
      "prediction": 0.9144989252090454,
      "threshold": 0.5
    }
  ],
  "scores": [
    0.9144989252090454
  ]
}
```

2. Using Python Requests 

```
import requests, json

BASE = "http://localhost:9696"

# single-record prediction
sample = {
  "age":60,
  "anaemia":0,
  "creatinine_phosphokinase":250,
  "diabetes":0,
  "ejection_fraction":38,
  "high_blood_pressure":0,
  "platelets":260000,
  "sex":1,
  "serum_creatinine":1.1,
  "serum_sodium":137,
  "smoking":0,
  "time":45
  }
response = requests.post(url, json=data)
print(response.json())
```
Expected Response: 
```
{
  "model_metrics": {
    "accuracy": 0.992,
    "roc_auc": 0.9994259853128142,
    "threshold": 0.5
  },
  "results": [
    {
      "class_label": 1,
      "heart_failure": true,
      "prediction": 0.9144989252090454,
      "threshold": 0.5
    }
  ],
  "scores": [
    0.9144989252090454
  ]
}
```

### Future Improvements

1. Deploy model in any cloud 
2. Explore deep learning models for complex feature interactions.

Scalability:

- Deploy on Kubernetes to handle high traffic and ensure availability.

User Interface:

- Add a web-based dashboard.












