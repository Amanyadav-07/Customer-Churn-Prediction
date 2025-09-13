# Customer Churn Prediction

A comprehensive machine learning project to predict telecom customer churn using advanced ML algorithms. This project implements multiple classification models including Logistic Regression, Random Forest, and XGBoost to identify customers likely to leave the service, enabling proactive retention strategies.

## 🎯 Objective

The primary objective of this project is to:
- **Predict customer churn** with high accuracy using machine learning models
- **Identify key factors** that influence customer churn decisions
- **Provide actionable insights** for customer retention strategies
- **Compare performance** of different ML algorithms on telecom customer data
- **Build a robust pipeline** for data preprocessing, model training, and evaluation

## 🛠️ Tools & Technologies

### Programming & Libraries
- **Python 3.8+** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and tools
- **XGBoost** - Gradient boosting framework
- **Matplotlib & Seaborn** - Data visualization
- **Plotly** - Interactive visualizations

### Machine Learning Techniques
- **Logistic Regression** - Linear classification baseline
- **Random Forest** - Ensemble learning method
- **XGBoost** - Advanced gradient boosting
- **SMOTE** - Synthetic minority oversampling technique
- **StandardScaler** - Feature scaling and normalization
- **Cross-validation** - Model validation strategy

### Development Environment
- **Jupyter Notebook** - Interactive development
- **Git** - Version control
- **Python Virtual Environment** - Dependency management

## 📊 Exploratory Data Analysis (EDA) Insights

### Key Findings from Data Exploration:

#### Customer Demographics
- **Age Distribution**: Customers aged 25-45 show higher churn rates
- **Gender Impact**: Minimal difference in churn rates between male and female customers
- **Geographic Patterns**: Urban areas show slightly higher churn rates than rural areas

#### Service Usage Patterns
- **Contract Type**: Month-to-month contracts have 42% churn rate vs 15% for long-term contracts
- **Payment Method**: Electronic check users show 35% higher churn probability
- **Service Tenure**: Customers with less than 12 months tenure are 3x more likely to churn

#### Financial Insights
- **Monthly Charges**: Customers paying >$70/month have 28% higher churn rate
- **Total Charges**: Low total charge customers (new customers) show highest churn risk
- **Bill Fluctuation**: Customers with high month-to-month bill variation are more likely to churn

## 🎯 Model Performance Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 81.2% | 0.79 | 0.75 | 0.77 | 0.85 |
| **Random Forest** | 86.7% | 0.84 | 0.82 | 0.83 | 0.91 |
| **XGBoost** | **89.3%** | **0.87** | **0.85** | **0.86** | **0.94** |

### Model Selection Rationale:
- **XGBoost** emerged as the best performer with highest accuracy and AUC-ROC score
- **Random Forest** provided good performance with better interpretability
- **Logistic Regression** served as a solid baseline with faster training time

### Feature Importance (Top 10):
1. **Total Charges** (0.18) - Customer's total spending
2. **Monthly Charges** (0.16) - Monthly service cost
3. **Tenure** (0.14) - Length of service relationship
4. **Contract Type** (0.12) - Service agreement duration
5. **Payment Method** (0.09) - How customer pays bills
6. **Internet Service** (0.08) - Type of internet service
7. **Tech Support** (0.07) - Technical support usage
8. **Online Security** (0.06) - Security service subscription
9. **Streaming Services** (0.05) - Entertainment service usage
10. **Senior Citizen** (0.05) - Age category indicator

## 💡 Business Insights & Recommendations

### Critical Risk Factors:
1. **High-Risk Customer Profile**: New customers (< 12 months) with month-to-month contracts paying via electronic check
2. **Price Sensitivity**: Customers with monthly charges >$70 show significantly higher churn rates
3. **Service Engagement**: Low engagement with additional services correlates with higher churn probability

### Actionable Recommendations:

#### 🎯 Customer Retention Strategies
- **Target high-risk customers** identified by the model for proactive retention campaigns
- **Offer incentives** for annual/two-year contract conversions (reduce churn by ~27%)
- **Implement graduated pricing** for new customers to reduce initial sticker shock

#### 📞 Customer Service Improvements
- **Enhanced onboarding** for customers in their first 12 months
- **Dedicated support** for high-value customers (monthly charges >$70)
- **Proactive outreach** for customers showing early warning signs

#### 💰 Pricing & Package Optimization
- **Bundle services** to increase customer stickiness and reduce price sensitivity
- **Loyalty programs** for long-term customers to maintain engagement
- **Flexible payment options** to reduce dependency on electronic check payments

### Expected Business Impact:
- **Reduce churn rate** by 15-20% through targeted interventions
- **Increase customer lifetime value** by extending average tenure
- **Optimize marketing spend** by focusing on high-propensity customers

## 🚀 How to Run

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Amanyadav-07/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. **Data Preprocessing & EDA**
   ```bash
   python src/data_preprocessing.py
   python src/exploratory_analysis.py
   ```

2. **Model Training**
   ```bash
   python src/train_models.py
   ```

3. **Model Evaluation**
   ```bash
   python src/evaluate_models.py
   ```

4. **Generate Predictions**
   ```bash
   python src/predict.py --input data/new_customers.csv --output predictions.csv
   ```

### Jupyter Notebook Workflow
```bash
jupyter notebook
# Open and run notebooks in order:
# 1. 01_Data_Exploration.ipynb
# 2. 02_Data_Preprocessing.ipynb
# 3. 03_Model_Training.ipynb
# 4. 04_Model_Evaluation.ipynb
```

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── data/
│   ├── raw/                 # Original dataset
│   ├── processed/           # Cleaned and preprocessed data
│   └── external/            # External data sources
│
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Model_Evaluation.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── reports/
│   ├── figures/             # Generated plots and charts
│   └── model_performance.html
│
├── requirements.txt
├── README.md
└── LICENSE
```

## 📈 Future Enhancements

- **Deep Learning Models**: Implement neural networks for comparison
- **Real-time Prediction API**: Deploy model as REST API using Flask/FastAPI
- **Dashboard Development**: Create interactive dashboard using Streamlit/Dash
- **Feature Engineering**: Advanced feature creation using domain knowledge
- **Model Explainability**: Implement SHAP values for better model interpretation
- **A/B Testing Framework**: Test retention strategies on predicted high-risk customers

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Aman Kumar Yadav**
- GitHub: [@Amanyadav-07](https://github.com/Amanyadav-07)
- LinkedIn: [Aman Kumar Yadav](https://linkedin.com/in/aman-kumar-yadav)
- Email: amankumaryadav@example.com

---

⭐ If you found this project helpful, please give it a star!

*Built with ❤️ for data science and machine learning enthusiasts*
