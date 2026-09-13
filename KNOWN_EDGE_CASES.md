# Known Edge Cases & Governance Framework

## 1. Boundary Vulnerabilities
* **Issue:** Ingestion of zero or negative vehicle prices / horsepower values.
* **Risk:** Distorts linear regression coefficient calculation and evaluation metrics.
* **Safeguard:** `clean_and_validate_data()` enforces strict strict lower bounds (`> 0`) and throws an explicit `ValueError`.

## 2. Structural & Missing Data
* **Issue:** Datasets containing nulls (`NaN`) or zero records.
* **Risk:** Pipeline crash during matrix operations in scikit-learn.
* **Safeguard:** Ingestion checks verify DataFrame length prior to execution and safely drop missing fields.

## 3. Data Type Safety
* **Issue:** Unstructured string inputs inside numeric metric fields.
* **Risk:** Unhandled Python runtime exceptions.
* **Safeguard:** Explicit `TypeError` assertions validate data schema types before model fitting.