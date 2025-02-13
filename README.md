# 🏡 Boston House Price Prediction

This project predicts house prices in Boston using machine learning.

## 📂 Dataset
- Source: [Boston Housing Dataset](https://www.kaggle.com/datasets)
- Features: Various property attributes
- Target: `MEDV` (Linear Regression Model)

## 🛠 Technologies Used
- Python 🐍
- Pandas & NumPy (Data Processing)
- Scikit-Learn (Machine Learning)
- Matplotlib & Seaborn (Data Visualization)
- Statsmodels (Linear Regression Analysis)

## 📊 Model Performance

### **🔹 Key Metrics**
| Metric | Value |
|--------|-------|
| **R² Score** | 0.714 |
| **Adjusted R²** | 0.709 |
| **F-statistic** | 161.1 (p < 0.0001) |
| **AIC (Model Complexity)** | -82.63 |
| **BIC (Bayesian Information Criterion)** | -54.78 |

### **🔹 Feature Impact on House Prices**
| Feature | Coefficient (`coef`) | Effect on Price |
|---------|----------------|----------------|
| **LSTAT (Low-income %)** | -0.0372 | 🔻 Higher poverty rate → Lower prices |
| **CRIM (Crime Rate)** | -0.0103 | 🔻 Higher crime → Lower prices |
| **DIS (Distance to Jobs)** | -0.0410 | 🔻 Farther from jobs → Lower prices |
| **CHAS (Near Charles River)** | +0.1421 | 🔺 Near the river → Higher prices |
| **ZN (Residential Land %)** | +0.0029 | 🔺 More land → Higher prices |
| **RAD (Highway Access)** | -0.0043 | 🔻 More highways → Slight price decrease |

### **📌 Interpretation**
- The model **explains 71.4%** of house price variation (**R² = 0.714**).
- **Lower-income population (LSTAT)** has the **strongest negative effect** on price.
- **Houses near the Charles River (CHAS)** tend to be more expensive.
- **Crime rate (CRIM) significantly reduces house prices**.

## 🚀 How to Run the Project
1. Install required libraries:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn statsmodels
   
2. Run the Script:
   ```bash
   python boston_house_price_prediction.py

3. The model will train and show results.

## 📝 Author
Ruben Treviño

GitHub: Aluches
   