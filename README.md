# Motor Insurance Pricing — Frequency-Severity Actuarial Model

## Overview
This project builds a professional actuarial pricing model for motor insurance using the classical **Frequency-Severity** methodology, benchmarked against modern machine learning approaches (XGBoost/LightGBM). It demonstrates the full pipeline from raw claims data to a commercial premium, including an interactive pricing dashboard.

## Business Problem
Insurance companies need to price policies fairly: charge enough to cover expected claims and expenses, while remaining competitive. This project answers: **"What premium should we charge a given policyholder, based on their risk profile?"**

## Methodology

1. **Frequency Model** — Poisson / Negative Binomial GLM predicting the expected number of claims per policyholder, based on features like age, vehicle type, region, and driving experience.
2. **Severity Model** — Gamma GLM predicting the average cost per claim, given that a claim occurs.
3. **Pure Premium** — `Pure Premium = Expected Frequency × Expected Severity`
4. **ML Benchmark** — XGBoost/LightGBM models compared against the GLMs, discussing the interpretability vs. accuracy trade-off relevant to regulated insurance pricing.
5. **Commercial Premium** — Pure premium adjusted for expenses, profit margin, and reinsurance loading to produce the final price charged to customers.
6. **Interactive Dashboard** — A Streamlit app where a user inputs a policyholder profile and receives an instant premium quote.

## Dataset
[French Motor Third-Party Liability (freMTPL2)](https://www.kaggle.com/datasets/karansarpal/fremtpl2-french-motor-tpl-insurance-claims) — a well-known dataset in actuarial research and education (originally from the CASdatasets R package), containing ~678,000 motor insurance policies with claim frequency and severity data.

## Project Structure
```
motor-insurance-pricing/
├── data/
│   ├── raw/                  # Original freMTPL2freq.csv and freMTPL2sev.csv
│   └── processed/            # Cleaned, merged dataset
├── notebook/
│   └── motor_pricing_project.ipynb   # Full analysis pipeline
├── dashboard/
│   └── app.py                # Streamlit pricing dashboard
├── reports/
│   └── final_report.pdf      # Summary report for non-technical stakeholders
├── requirements.txt
└── README.md
```
## Interactive dashoard URL
motor-insurance-pricing-hdusnshbnhqsr6gfnrm7is.streamlit.app/

## Key Results
*(To be filled in after model training — e.g. model performance metrics, top risk factors, pricing examples)*

## Tech Stack
- Python (pandas, numpy, statsmodels, scikit-learn)
- XGBoost / LightGBM
- Streamlit (dashboard)
- Matplotlib / Seaborn / Plotly (visualization)

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook/motor_pricing_project.ipynb
streamlit run dashboard/app.py
```

## Author
Abdellah El Khamlichi — Master's in Finance, Actuarial Science & Data Science

## License
MIT
