# Car Price Prediction & Data Pipeline

A production-grade machine learning data pipeline and automated testing framework designed to process, clean, and validate automotive datasets for price regression modeling.

## Project Architecture

Unlike traditional exploratory notebooks that tightly couple data loading, cleaning, and modeling, this project adopts a **modular architecture** to separate concerns and ensure maintainability:

* **`src/data_pipeline.py`**: The core logic file containing the data ingestion, type-safety enforcement, missing-value handling, and boundary verification logic. Acts as a single source of truth for both experimental notebooks and production code.
* **`tests/test_data_pipeline.py`**: A comprehensive automated test suite leveraging `pytest` to guarantee data integrity.
* **Jupyter Notebooks (`*.ipynb`)**: Utilizes the modular pipeline function to execute exploratory data analysis and train regression models seamlessly.

---

## Automated Testing Suite

To protect downstream machine learning models from data poisoning, structural corruption, and silent runtime failures, a strict test suite of **14 automated unit tests** was engineered using `pytest`. 

The test cases cover:
1. **Happy Path & Boundaries:** Validating standard clean inputs and edge limits (e.g., minimum valid horsepower and price thresholds).
2. **Data Hygiene & Nulls:** Verifying the robust handling and dropping of missing (`NaN`) values.
3. **Business Logic & Error Handling:** Enforcing strict domain rules and auditing error messages for negative/zero values or empty datasets.
4. **Type Safety & Schema Protection:** Safeguarding against type errors by explicitly blocking non-numeric inputs (such as text strings or boolean flags which Pandas natively treats as numbers).

### Running the Tests

To execute the test suite and verify system health:

```bash
python -m pytest -v
