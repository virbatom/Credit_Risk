# Credit Risk Analysis: Portfolio Insights & Default Prediction

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14+-3D5A80?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

A full end-to-end credit risk analysis on a historical loan portfolio of
10,000 clients. The project identifies the drivers of default, evaluates the
profitability of each distribution channel, and builds a logistic regression
model to classify defaulters. Covering the full
data-science workflow: EDA, statistical inference, channel economics, and
predictive modelling with ROC analysis.

> **Author:** Tomáš Virba — Econometrics & Operations Research, Prague
> University of Economics and Business (VŠE)

---

## Project at a Glance

| Detail | Value |
|---|---|
| **Dataset size** | 10,000 clients |
| **Loan amount** | 30,000 (uniform across all clients) |
| **Default rate** | 5.3% |
| **Distribution channels** | Website, Facebook, Broker |
| **Model** | Logistic Regression with dummy-encoded categoricals |
| **🏆 Best result** | **ROC AUC = 0.88** (Logistic Regression, 95.7% accuracy) |

---

## Objective

Analyse a loan portfolio of 10,000 clients with three goals:

1. **Understand who defaults** — quantify how demographics (age, gender,
   marital status, housing, education, income, children) relate to default
   risk
2. **Evaluate channel economics** — measure profitability of the website,
   Facebook, and broker acquisition channels under realistic recovery
   assumptions
3. **Build a predictive model** — train a logistic regression classifier and
   discuss the trade-off between precision and recall in a credit-risk
   setting

---

## Repository Structure

```
Credit_Risk/
├── data/
│   ├── historical_sample.csv     # 10k client records
│   └── broker_list.csv           # broker-channel clients
├── Credit_Risk.ipynb             # Main notebook (analysis + modelling)
├── Credit_Risk.html              # Rendered notebook (view without running)
└── README.md
```

---

## Dataset

The dataset contains 10,000 loan records with demographic, socio-economic, and
behavioural variables. Each client received the same loan amount of 30,000.

| Variable | Description |
|---|---|
| `Age` | Client age (18–65) |
| `Gender` | 0 = Male, 1 = Female |
| `MarStat` | Marital status (1 = Single, 2 = Married, 3 = Divorced, 4 = Widowed) |
| `Housing` | 1 = With parents, 2 = Town hall, 3 = Own house, 4 = Rented |
| `Educ` | 1 = Elementary, 2 = High school, 3 = University, 4 = Postgraduate |
| `Inc_Gr` | 1 = Social security, 2 = < 15k, 3 = 15–30k, 4 = > 30k |
| `Children` | Number of children (0–3) |
| `IQ` | IQ score |
| `rate` | Interest rate on the loan |
| `cost_type` | Distribution channel (www, Facebook, Broker) |
| `Default_flg` | 1 = Defaulted, 0 = Repaid |

---

## Methodology

### 1. Data preparation
- Fixed European decimal format (comma → dot) on `IQ` and `rate`
- Encoded distribution channel numerically (www=1, Facebook=2, Broker=3)
- Created 5-year age bins for contingency analysis

### 2. Portfolio overview & LGD
**Loss Given Default** computed using housing type as a proxy for collateral:
- **20%** LGD for homeowners (`Housing = 3`) — can offer collateral
- **70%** LGD for all other housing types — no collateral

### 3. Channel selection analysis
Discovered a striking pattern: the channel is **almost entirely determined by
age**. Verified through three lenses:
- **Contingency tables** — brokers used only by clients aged 38+, website by
  under-38, Facebook by a narrow 33–42 band
- **OLS regression** — R² = 0.743, but high condition number (633) flagged
  multicollinearity. Log-transformation cut it to 284
- **Multinomial logistic regression** — Age + IQ → **99.6% accuracy**;
  Age alone → **96.5%**

### 4. IQ ↔ Age relationship
- Pearson correlation: **0.99** (near-perfect linear dependence)
- OLS: R² = 0.971, slope ≈ 2.84 IQ points per year of age
- This collinearity is the reason Age alone suffices for channel prediction

### 5. Default rate vs. income
- Default falls monotonically from **10.6%** (social security) to **2.0%**
  (income > 30k)
- ANOVA confirms group differences (p ≪ 0.001)
- Pearson r = -0.15 — modest only because the target is binary

### 6. Channel profitability
Per-client cash flow:

```
cash = realised_repayment − (loan + channel_cost)
```

with channel costs of **200 (www), 300 (Facebook), 1,000 (Broker)** and
recovery rates of **80% for homeowners / 30% otherwise** on defaulted loans.

### 7. Best-vs-worst client profiles
- **Best clients** = top 10% by cash flow
- **Worst clients** = any client with `cash < 0`
- Compared normalised frequency distributions across `Gender`, `MarStat`,
  `Housing`, `Educ`, `Inc_Gr`, `Children`, age group, and channel

### 8. Default prediction model
**Logistic regression** with L2 penalty, threshold = 0.4, on dummy-encoded
categorical features (`MarStat`, `Housing`, `Educ`, `Inc_Gr`, `cost_type`)
plus numeric variables (`Age`, `Gender`, `Children`, `IQ`, `rate`).
OLS is included as a baseline to show why linear regression is inappropriate
for binary outcomes (R² = 0.084).

---

## Results

### Model performance

| Metric | Value |
|---|---|
| **ROC AUC** | **0.88** |
| Accuracy | 95.7% |
| Precision | high |
| Recall (Sensitivity) | 17.5% |
| Specificity | very high |

The AUC of **0.88** indicates strong discriminatory power — the model ranks a
random defaulter above a random non-defaulter 88% of the time.

### The precision–recall trade-off

The confusion matrix exposes the same asymmetry real banks fight over
constantly:

- **High specificity, low sensitivity** — the model catches only **7 of 40**
  actual defaulters in the test set
- Whether this is acceptable depends entirely on the **cost of a missed
  default vs. the revenue from an extra approved loan**
- Lowering the 0.4 threshold would trade some precision for considerably
  better recall — a decision that belongs to the credit committee, not the
  data scientist

### Channel profitability

All three channels are currently **unprofitable**:
- **Broker** — crushed by the 1,000-per-client acquisition cost
- **Website** — largest total loss simply because it serves the most clients
- **Facebook** — smallest average loss (-54.91) but tiny sample (752 clients)

---

## Tech Stack

- **Python 3.10+**
- **pandas**, **NumPy** — data manipulation
- **matplotlib**, **seaborn** — visualization
- **scipy.stats** — statistical tests (ANOVA, correlation)
- **statsmodels** — OLS regression with full diagnostics (Durbin–Watson,
  Omnibus, condition number)
- **scikit-learn** — logistic regression, train/test split, ROC, metrics

---

## How to Run

### Option 1 — view the rendered notebook (no setup)
Open [`Credit_Risk.html`](Credit_Risk.html) in any browser.

### Option 2 — run the notebook locally

```bash
# 1. Clone the repository
git clone https://github.com/virbatom/Credit_Risk.git
cd Credit_Risk

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn jupyter

# 4. Launch Jupyter
jupyter notebook Credit_Risk.ipynb
```

Random seeds are fixed (`random_state=42` / `421`) so results are reproducible.

---

## 📝 Key Takeaways

1. **Channel selection is almost entirely an age story.** Age and IQ are
   correlated at 0.99 in this dataset, and either one predicts the chosen
   channel with > 96% accuracy. Older clients go to brokers, younger ones
   stay on the website.
2. **Income matters — but not in isolation.** Default rates fall from 10.6%
   to 2.0% across income groups, yet income alone explains only R² = 0.023.
   Useful inside a multivariate model, weak as a standalone signal.
3. **The portfolio is currently loss-making on every channel.** The broker
   channel is killed by acquisition costs; the website bleeds slowly through
   sheer volume.
4. **Best-vs-worst profiles are sharply different.** Best performers are
   typically female, 28–37, university-educated, homeowners, high income,
   acquired via the website. Worst performers are typically male, only
   elementary-educated, living with parents, on social security, acquired via
   brokers.
5. **AUC ≠ recall.** The model achieves 95.7% accuracy and AUC = 0.88 but
   catches only 17.5% of real defaulters. Threshold tuning and
   cost-sensitive learning are the obvious next steps.

### Practical recommendations

- **Tighten approvals** until the portfolio returns to breakeven
- **Reconsider the 5% interest rate** — may be too low for the actual risk
- **Negotiate broker fees** or shift acquisition toward cheaper digital
  channels
- **Tune the decision threshold** to match the bank's risk appetite, not the
  default 0.5

---

## 📬 Contact

Built by **Tomáš Virba** 

- GitHub: [@virbatom](https://github.com/virbatom)

If you find issues or have suggestions, feel free to open an Issue or PR.

---

*This project was developed for educational and portfolio purposes. It is not
intended for production credit-scoring use.*
