# 🩺 Healthcare Cost Prediction Using Neural Networks

This project predicts **healthcare insurance costs** using a **Conv1D neural network** built with TensorFlow and Keras.  
By combining deep learning, feature engineering, and explainable AI (via SHAP), it identifies the most important factors driving healthcare expenses.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Overview

Healthcare costs vary based on several personal and lifestyle factors such as age, BMI, smoking status, and region.  
This project uses a deep learning approach to accurately estimate those costs, achieving an **R² score of 0.88** on test data.

### Key Highlights
- **Conv1D Neural Network** optimized for structured data  
- **Feature Engineering** with interaction and polynomial terms  
- **SHAP Analysis** for transparent, interpretable predictions  
- **Performance**:  
  - R² = 0.8778  
  - RMSE = 4355.09  
  - MAE = 2619.89  
  - MAPE = 0.3310  

---

## 🧠 Model Architecture

```

Conv1D(64 filters, kernel_size=2) → Flatten
↓
Dense(128, relu) → Dropout(0.2)
↓
Dense(64, relu)
↓
Dense(32, relu)
↓
Dense(1, linear)

```

The model uses a 1D convolution layer followed by several dense layers with dropout regularization.  
It’s trained using the Adam optimizer and Mean Squared Error (MSE) loss.

---

## 📂 Project Structure

```

healthcare-cost-prediction/
│
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
├── .gitignore             # Ignore rules for Git
│
├── data/                  # Dataset folder (not tracked)
│   └── .gitkeep
│
├── notebooks/             # Jupyter notebooks
│   └── healthcare-cost-prediction.ipynb
│
├── src/                   # Source code
│   ├── data_processing.py
│   ├── model.py
│   └── utils.py
│
└── images/                # Visualizations
└── .gitkeep

````

---

## 📦 Getting Started

### Prerequisites
```bash
Python 3.10+
pip
````

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/RYANX9/healthcare-cost-prediction.git
cd healthcare-cost-prediction
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

This project uses the public **Insurance Dataset** from Kaggle:
🔗 [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)

Download the dataset from Kaggle and place the file `insurance.csv` inside the `data/` directory.

```
data/
└── insurance.csv
```

---

## 💻 Usage

### Run the notebook

```bash
jupyter notebook notebooks/healthcare-cost-prediction.ipynb
```

### Quick Prediction Example

```python
from src.model import predict_healthcare_cost

# Example prediction
cost = predict_healthcare_cost(
    age=30,
    bmi=31.2,
    children=0,
    smoker='no',
    region='northeast',
    sex='male'
)
print(f"Predicted cost: ${cost:.2f}")
```

---

## 🔍 Insights & Findings

1. **Smoking status** has the largest impact on healthcare costs.
2. **Age and BMI** show a strong combined effect on insurance charges.
3. **Feature engineering** improved the model’s accuracy by around 15%.

---

## 📈 Visualizations

The project includes several useful plots:

* Training and validation loss curves
* Predicted vs. actual cost scatter plots
* SHAP feature importance
* Error distribution visualization

---

## 🛠️ Technologies Used

* **Deep Learning**: TensorFlow, Keras
* **Data Processing**: Pandas, NumPy, Scikit-learn
* **Visualization**: Matplotlib, Seaborn, Plotly
* **Explainability**: SHAP

---

## 🤝 Contributing

Contributions are always welcome!
Follow these steps to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for more details.

---

## 👤 Author

**Ahmed Messaad**

* GitHub: [@RYANX9](https://github.com/RYANX9)
* LinkedIn: [Ahmed Messaad](https://linkedin.com/in/ahmedmessaad)

---

## 🙏 Acknowledgments

* Dataset: [Kaggle – Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
* TensorFlow and Keras documentation
* Inspiration from the healthcare analytics community

---

## 📧 Contact

For questions, suggestions, or collaborations, please open an issue or contact me at **[ahmed.messaad@outlook.com](mailto:ahmed.messaad@outlook.com)**.

---

⭐ **If you found this project helpful, please consider giving it a star on GitHub!**

```
```
