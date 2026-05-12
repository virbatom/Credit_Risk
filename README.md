# Credit Risk Analysis: Portfolio Insights & Default Prediction

**Author:** Tomas Virba  
**Goal:** Analyze a historical loan portfolio of 10,000 clients to identify drivers of default, evaluate channel profitability, and build a predictive logistic regression model.

---

##  Project Overview
This project examines a loan portfolio where each client received a uniform loan amount of 30,000. Despite a relatively low default rate of 5.3%, the portfolio is currently unprofitable across all distribution channels. This analysis identifies the demographic markers of high-risk clients and provides a model to mitigate future losses.

### Key Metrics
| Detail | Value |
| :--- | :--- |
| **Dataset Size** | 10,000 clients |
| **Default Rate** | 5.3% |
| **Loan Amount** | 30,000 (uniform) |
| **Best Model AUC** | 0.88 |

---

##  Data Insights

### 1. The Age-IQ-Channel Connection
The analysis revealed a near-perfect correlation (**0.99**) between Age and IQ. This dependency extends to how clients choose their loan provider:
* **Website:** Preferred by younger clients (under 38).
* **Facebook:** Used by a narrow middle-age range (33-42).
* **Brokers:** Exclusively used by clients aged 38 and above.

### 2. Profitability & Recovery
Currently, all channels show a net loss per client:
* **Broker Channel:** Suffers from high acquisition costs (1,000 per client).
* **Website:** Carries the largest total loss due to high volume and negative average margins.
* **Loss Given Default (LGD):** Recovery is significantly higher for homeowners (80%) compared to other housing types (30%).

### 3. Client Profiles
* **Best-performing clients:** Often female, aged 28-37, university-educated homeowners with high income.
* **Worst-performing clients:** Often male, elementary education, living with parents or registered at town hall, on social security.

---

##  Modeling & Performance
A **Logistic Regression** model was trained using features such as Age, Gender, Education, and Income Group.

* **Accuracy:** 95.7% (at a 0.4 threshold).
* **AUC Score:** 0.88, indicating strong discriminatory power.
* **Sensitivity (Recall):** 17.5%. The model is conservative and prioritizes avoiding false alarms over catching every defaulter.

---

## 🚀 Recommendations
* **Tighten Approvals:** Focus lending on high-probability repayment profiles (homeowners, higher education).
* **Cost Reduction:** Negotiate lower broker fees or shift acquisition toward cheaper digital channels like the website.
* **Pricing Strategy:** Re-evaluate the 5% interest rate to better reflect the current risk environment.
* **Model Tuning:** Lower the classification threshold to increase the detection of defaulters (Sensitivity) based on the bank's risk tolerance.

---

##  Tech Stack
* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`
