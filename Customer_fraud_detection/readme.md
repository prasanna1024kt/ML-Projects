# Customer Fraud Detection - (CIS)

## Dataset Details
The IEEE-CIS (Computational Intelligence Society) Fraud Detection dataset contains two core tables:

| Dataset | Rows | Columns | Key Column | Description |
|---|---:|---:|---|---|
 File Name | Rows | Key Column | Fraud Label | Description |
| **train_transaction.csv** | 590,540 | `TransactionID` | `isFraud`  | Transactional and engineered behavioral features for training |
| **train_identity.csv** | 144,233 | `TransactionID` | `isFraud`  | Device, browser, network, and digital fingerprint signals for train set |
| **test_transaction.csv** | 506,691 | `TransactionID` | `isFraud` | Transaction features for model evaluation/prediction |
| **test_identity.csv** | 141,907 | `TransactionID` | `isFraud` | Identity signals corresponding to test transactions |

### Joining Strategy

Identity tables should be left-joined to transaction tables separately for train and test:

```python
df_train = df_trans_train.merge(df_id_train, on="TransactionID", how="left")
df_test  = df_trans_test.merge(df_id_test, on="TransactionID", how="left")
```
###  Label Availability

- isFraud is available only in train_transaction.csv

- Test tables do not contain the fraud label, as they are intended for generating model predictions.

Feature Consistency

Train and test share the same schema and feature groups, including:

- Card: card1–card6, card4, card6

- Counters: C1–C14

- Time deltas: D1–D15

- Match flags: M1–M9

- Engineered signals: V1–V339

- Identity: id_01 – id_38, DeviceType, DeviceInfo, id_31 (Browser)

- Risk signals: TransactionAmt, P_emaildomain, R_emaildomain, addr1,  addr2, dist1, dist2


### Data Characteristics
- Mixed dtypes: numeric (`float`, `int`) + categorical (`string`, `email domain`, `flags`)
- Transaction time is stored in `TransactionDT` (seconds from dataset start, not actual timestamp)
- Target label: `isFraud` (1 = Fraud, 0 = Non-Fraud)

---

## Key Objective
Detect fraudulent e-commerce transactions using behavioral and identity signals to improve risk scoring, prevent fraud, and optimize automated decision systems.

---

## Important Feature Groups & Columns

### Payment & Card Details
| Column | Meaning | Importance |
|---|---|---|
| `TransactionAmt` | Transaction amount | Outlier and risk indicator |
| `ProductCD` | Product category code | Segment-wise fraud distribution |
| `card1–card6` | Card metadata (bank, type, category, issuer, country, etc.) | Strong fraud linkage |
| `card4` | Card network (Visa, Mastercard, Amex, Discover) | Network-based fraud rate |
| `card6` | Debit/Credit type | Payment risk pattern |

### Time & Behavioral
| Column | Meaning | Importance |
|---|---|---|
| `TransactionDT` | Time delta in seconds | Used to derive fraud spikes (hour/day) |
| `D1–D15` | Transaction time deltas | Behavioral risk patterns |
| `C1–C14` | Counting features | Frequency signals for fraud |
| `M1–M9` | Match flags | Transaction consistency indicators |
| `V1–V339` | Engineered features | Most predictive fraud indicators |

### Digital Identity Signals (from identity table)
| Column | Meaning | Importance |
|---|---|---|
| `DeviceType` | Mobile/Desktop | Device segment fraud rate |
| `DeviceInfo` | Hardware/OS details | Fingerprint risk indicator |
| `id_31` | Browser signature | Browser-based fraud link |
| `id_33` | Screen resolution | Device spoofing detection |
| `id_01–id_38` | Anonymized identity features | Digital fingerprint and network risk |

### Email & Address Risk
| Column | Meaning | Importance |
|---|---|---|
| `P_emaildomain` | Purchaser email domain | Domain risk scoring |
| `R_emaildomain` | Receiver email domain | Fraud linkage check |
| `addr1, addr2` | Region/Country | Geo risk segmentation |
| `dist1, dist2` | Distance between transactions | Anomaly detection |

---

## Data Quality Observations
Common issues found in CIS fraud data:
- High missing % in many `V` and `id_` columns (expected due to anonymization and sparse feature generation)
- Presence of duplicate row patterns should be checked using `TransactionID`
- Corrupted rows originate from CSV parsing errors, not from Parquet conversion — validate before modeling
- Left join increases missing values in identity-based columns when no identity match exists (expected)

---

## EDA Insights Covered in the Report
The EDA pipeline analyzes:

1. **Fraud % across business segments** (`ProductCD`, `card4`, `card6`, `addr1/addr2`)
2. **Fraud spikes over time** (derived `hour` and `day` from `TransactionDT`)
3. **Device & browser fraud footprint** (`DeviceType`, `DeviceInfo`, `id_31`)
4. **Top correlated numeric fraud indicators** (from `V`, `C`, `D` feature groups)
5. **Dataset integrity issues** (missing %, duplicates, high-null rows)

---

## How to Use the Dataset 
Example (after downloading from Kaggle):

```python
df_trans = pd.read_parquet("train_transaction.parquet", low_memory=False)
df_id = pd.read_parquet("train_identity.parquet", low_memory=False)
df_merge = df_trans.merge(df_id, on="TransactionID", how="left")
df_test_trans = pd.read_parquet("test_transaction.parquet", low_memory=False)
df_test_id = pd.ead_parquet("test_identity.parquet", low_memory=False)
df_test_merge = df_test_trans.merge(df_test_id, on="TransactionID", how="left")
```

## Repo layout
- `Customer_fraud_detection/` — project root
- `datasets/` — place `train_transaction.parquet`, `train_identity.parquet`, `test_transaction.parquet`, `test_identity.parquet` here
- `model/` — training & serving code, saved models, predictions
  - `train.py` — preprocessing, training, evaluation, model export
  - `predict.py`  — Flask prediction service
  - `lgb_model.bin` — saved model artifacts
  - `predictions.csv` — example output
- `Dockerfile` — container image for the service
- `requirements.txt` or `Pipfile` — Python dependencies
- `notebooks/` — EDA analysis , conversion csv to parquet file,python request , 

## Instructions to Run the Application

### Prerequisites
- Python 3.13: Ensure Python is installed on your machine.
- Pipenv: Install Pipenv for dependency management.
- Docker: To run the application in a containerized environment.

### Local Setup 

#### 1. Clone repo 

```
git clone https://github.com/prasanna1024kt/ML-Projects.git
cd ML-Projects/Customer_fraud_detection
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
http://localhost:9696/predict
```

### Using Docker 

1. Build the Docker container:

``` docker build -t fraud-service:latest-pk -f Dockerfile . ```

2. Run the container:

``` docker run -p 9696:9696 fraud-service:latest-pk ```

### Test Deployment 

1. Using cURL Send a POST request to test the API:

```  
  curl -X POST http://localhost:9696/predict \
-H "Content-Type: application/json" \
-d '{
  "data": [
    {
      "TransactionID": 3663552,
      "TransactionAmt": 284.95,
      "card1": 10989,
      "C1": 5,
      "C2": 2.0,
      "C13": 7,
      "V12": 1,
      "hour": 0,
      "amt_log": 5.655817,
      "card_txn_count": 839
    }
  ]
}'

```
OutPut: 

``` 
{
  "TransactionID": [
    3663552
  ],
  "prediction": [
    0.8807577245894259
  ]
}
```

2. Using Python Requests 

```
import requests

url = "http://192.168.31.16:9696/predict"

payload = {
    "data": [
        {
            "TransactionID": 3663552,
            "TransactionAmt": 284.95,
            "card1": 10989,
            "C1": 5,
            "C2": 2.0,
            "C13": 7,
            "V12": 1,
            "hour": 0,
            "amt_log": 5.655817,
            "card_txn_count": 839
        }
    ]
}

response = requests.post(url, json=payload)
print(response.json())

```
Expected Response: 
```
{
  "TransactionID": [
    3663552
  ],
  "prediction": [
    0.8807577245894259
  ]
}

```
## API contract
- POST /predict
  - JSON body: {"data": [ {feature_dict}, ... ] } or {"data": {feature_dict}}
  - Response: {"TransactionID": [...], "prediction": [...]} (prediction = probability of fraud)
- GET /health — indicates whether model is loaded

## predict_test_dataset.py

A small CLI to run the saved model on the test parquet files and write predictions to CSV.

Usage
```bash
# from project root
python model/predict_test_dataset.py \
  --model model/lgb_model.bin \
  --datasets ./datasets/test_transaction.parquet \
  --datasets ./datasets/test_identity.parquet \
  --out model/predictions.csv
```

Defaults
- model: model/lgb_model.bin
- datasets: datasets/ (expects test_transaction.parquet and test_identity.parquet)
- out: model/predictions.csv

Behavior
- Loads test_transaction.parquet and test_identity.parquet, merges on TransactionID.
- Calls the same preprocess() used during training.
- Aligns features to the model (fills missing features with -999) before predicting.
- Writes CSV with columns: TransactionID,isFraud (probability).

## Troubleshooting
- macOS LightGBM errors: install OpenMP runtime:
  - brew install libomp
  - add to shell: export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH"
  - reinstall lightgbm in venv if needed.
- If Docker build fails due to platform-specific wheels (e.g. `contourpy`), build for amd64:
  - docker buildx build --platform linux/amd64 -t fraud-service:latest --load .
- If model expects N features but input has fewer, ensure:
  - You saved the model with feature names (use joblib) OR
  - You supply all model features (service will fill missing with -999, but order and names must match).
