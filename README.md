# 🌧️ Rainfall Prediction using Machine Learning & Streamlit

This project uses machine learning to predict the likelihood of rain the next day based on historical weather data. The system includes a web-based interface built with **Streamlit** for interactive and user-friendly predictions.

---

## 📂 Dataset Overview

The dataset contains **145,460 rows** and **23 features**, sourced from Australian weather observations. Some key features:

| Feature        | Description                                |
|----------------|--------------------------------------------|
| `MinTemp`      | Minimum temperature (°C)                   |
| `MaxTemp`      | Maximum temperature (°C)                   |
| `Rainfall`     | Amount of rainfall (mm)                    |
| `WindGustDir`  | Direction of strongest wind gust           |
| `WindSpeed9am` | Wind speed at 9 AM                         |
| `Humidity3pm`  | Humidity at 3 PM                           |
| `Pressure9am`  | Atmospheric pressure at 9 AM               |
| `Cloud3pm`     | Cloud cover at 3 PM                        |
| `RainToday`    | Whether it rained today (`Yes` / `No`)     |
| `RainTomorrow` | Target variable – rain next day (`Yes` / `No`) |

⚠️ Some features like `Evaporation`, `Sunshine`, and `Cloud` contain significant missing values and may need preprocessing or removal.

---

## 🧠 Machine Learning Model

- **Type**: Binary Classification
- **Goal**: Predict `RainTomorrow` (Yes/No)
- **Techniques Used**:
  - Label Encoding for categorical features
  - Imputation for missing values
  - Feature selection
  - Model: [e.g., Random Forest, Logistic Regression, XGBoost]
  - Evaluation Metrics: Accuracy, Precision, Recall, F1-Score

---

## 🌐 Streamlit Interface

The Streamlit app provides a web interface for users to:

- Input weather conditions manually
- View predictions instantly
- See probability scores (if applicable)

Run it with:

```bash
streamlit run app.py
