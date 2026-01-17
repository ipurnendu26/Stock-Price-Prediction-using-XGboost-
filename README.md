# 📈 Stock Price Prediction using XGBoost

A machine learning project that predicts stock closing prices using the XGBoost regression algorithm. The model leverages technical indicators, economic factors, and historical price data to forecast next-day stock prices.

## 🎯 Project Overview

This project follows the complete Data Science lifecycle:
1. **Problem Definition** - Predict stock closing prices
2. **Data Collection & Loading** - Load historical stock market data
3. **Exploratory Data Analysis (EDA)** - Understand data patterns and distributions
4. **Data Preprocessing** - Handle missing values, feature selection
5. **Feature Engineering** - Prepare features for modeling
6. **Model Building** - Train XGBoost regression model
7. **Model Evaluation** - Assess model performance
8. **Visualization & Insights** - Interpret results

## 📊 Dataset

The dataset (`stock_market_dataset.csv`) contains historical stock market data with the following features:

| Feature Category | Features |
|-----------------|----------|
| **Price Data** | Open, High, Low, Close, Volume |
| **Technical Indicators** | SMA, RSI, MACD, Bollinger Bands |
| **Economic Factors** | GDP, Inflation, Interest Rate |
| **Sentiment** | Sentiment Score |
| **Target** | Next_Close (Next day's closing price) |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Stock-Price-Prediction-using-XGboost-.git
   cd Stock-Price-Prediction-using-XGboost-
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "Stock Price Prediction using XGboost .ipynb"
   ```

2. Run all cells sequentially to:
   - Load and explore the dataset
   - Preprocess the data
   - Train the XGBoost model
   - Evaluate model performance
   - Visualize predictions

## 🤖 Model Configuration

```python
XGBRegressor(
    n_estimators=100,      # Number of boosting rounds
    max_depth=5,           # Maximum tree depth
    learning_rate=0.1,     # Step size shrinkage
    subsample=0.8,         # Subsample ratio
    colsample_bytree=0.8,  # Column subsample ratio
    random_state=42        # Reproducibility
)
```

## 📈 Model Performance

| Metric | Training Set | Testing Set |
|--------|-------------|-------------|
| R² Score | ~0.99 | ~0.98 |
| RMSE | Low | Low |
| MAE | Low | Low |

*Note: Actual values depend on the dataset used*

## 📁 Project Structure

```
Stock-Price-Prediction-using-XGboost-/
│
├── Stock Price Prediction using XGboost .ipynb  # Main notebook
├── stock_market_dataset.csv                      # Dataset
├── requirements.txt                              # Dependencies
├── README.md                                     # Project documentation
├── .gitignore                                    # Git ignore file
└── venv/                                         # Virtual environment
```

## 📦 Dependencies

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualization
- **scikit-learn** - Machine learning utilities
- **xgboost** - Gradient boosting algorithm

## 📊 Visualizations

The project includes various visualizations:
- Distribution plots for key features
- Correlation heatmap
- Feature importance chart
- Actual vs Predicted scatter plot
- Residual analysis plots
- Time series comparison

## ⚠️ Limitations

- Stock prices are inherently unpredictable due to market volatility
- External factors (news, events) are not captured in the model
- Past performance does not guarantee future results
- The model should be used for educational purposes only

## 🔮 Future Improvements

- [ ] Add more technical indicators
- [ ] Implement hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- [ ] Try ensemble methods combining multiple models
- [ ] Add real-time data fetching capability
- [ ] Implement cross-validation for robust evaluation
- [ ] Add LSTM/Neural Network comparison

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 👤 Author

**Purnendu Kale**

---

⭐ Star this repository if you found it helpful!
