# Healthcare Cost Prediction Using Neural Networks

A Conv1D neural network implementation for predicting healthcare insurance costs, achieving an R² score of 0.88. This project uses feature engineering and SHAP analysis to identify key cost drivers in healthcare prediction.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Project Overview

This project implements a deep learning solution for healthcare cost prediction using:
- **Conv1D Neural Network Architecture**: Optimal temporal convolution for sequential features
- **Feature Engineering**: Advanced feature creation including interactions and polynomial features
- **SHAP Analysis**: Explainable AI for understanding cost drivers
- **R² Score**: 0.88 on test set

## 📊 Key Features

- **Model Performance**:
  - R² Score: 0.8778
  - RMSE: 4355.09
  - MAE: 2619.89
  - MAPE: 0.3310

- **Architecture**:
  - Conv1D layer with 64 filters
  - Multiple Dense layers with Dropout
  - Adam optimizer with MSE loss

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.10+
pip
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/healthcare-cost-prediction.git
cd healthcare-cost-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset

Download the insurance dataset from [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) and place it in the `data/` directory.

## 📓 Usage

### Running the Notebook
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

## 📂 Project Structure
```
healthcare-cost-prediction/
│
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
│
├── data/                 # Dataset directory (not tracked)
│   └── insurance.csv
│
├── notebooks/            # Jupyter notebooks
│   └── healthcare-cost-prediction.ipynb
│
├── src/                  # Source code modules
│   ├── data_processing.py
│   ├── model.py
│   └── utils.py
│
└── images/               # Plots and visualizations
```

## 🔍 Key Findings

1. **Smoking Status**: Most significant predictor of healthcare costs
2. **Age & BMI Interaction**: Strong correlation with charges
3. **Feature Engineering**: Improved model performance by 15%

## 📈 Model Architecture
```
Conv1D(64 filters, kernel=2) → Flatten
    ↓
Dense(128, relu) → Dropout(0.2)
    ↓
Dense(64, relu)
    ↓
Dense(32, relu)
    ↓
Dense(1, linear)
```

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **ML Tools**: Scikit-learn
- **Explainability**: SHAP

## 📊 Results Visualization

The project includes comprehensive visualizations:
- Training/Validation loss curves
- Predictions vs Actual values scatter plots
- Feature importance analysis
- Error distribution plots

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: AHMED MESSAAD[@RYANX9](https://github.com/RYANX9)
- LinkedIn: AHMED MESSAAD(https://linkedin.com/in/ahmedmessaad)

## 🙏 Acknowledgments

- Dataset provided by [Kaggle Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Inspiration from healthcare analytics community
- TensorFlow and Keras documentation

## 📧 Contact

For questions or feedback, please open an issue or contact me at your.email@example.com

---

⭐ If you found this project helpful, please consider giving it a star!
```

### 4. `LICENSE` (MIT License)
```
MIT License

Copyright (c) 2024 Ahmed Messaad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 5. `data/.gitkeep` and `images/.gitkeep`
```
# This file ensures the directory is tracked by git
