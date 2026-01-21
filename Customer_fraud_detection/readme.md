# Customer Fraud Detection - (CIS)

## Dataset Details
The IEEE-CIS (Computational Intelligence Society) Fraud Detection dataset contains two core tables:

| Dataset | Rows | Columns | Key Column | Description |
|---|---:|---:|---|---|
 File Name | Rows | Key Column | Fraud Label | Description |
|---|---:|---|---|---|
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



