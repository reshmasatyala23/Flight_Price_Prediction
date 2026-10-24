# Flight Price Prediction Model

## Project Overview

The Flight Price Prediction Model is a machine learning project developed using Python, Linear Regression, and Streamlit. It predicts flight ticket prices based on relevant flight details.

The project uses Business and Economy flight datasets to explore flight pricing patterns and build a regression model for price prediction.

## Objectives

- Analyze flight pricing data.
- Understand the relationship between flight features and ticket prices.
- Build a machine learning model using Linear Regression.
- Evaluate model performance using regression metrics.
- Develop an interactive web application using Streamlit.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib / Seaborn (if used)

## Machine Learning Algorithm

**Linear Regression**

Linear Regression is a supervised machine learning algorithm used to predict continuous numerical values. It learns the relationship between independent variables (input features) and a dependent variable (flight ticket price).

The model estimates coefficients that minimize the squared differences between actual and predicted prices.

## Project Workflow

1. Load the Business and Economy flight datasets.
2. Explore and preprocess the data.
3. Prepare the input features and target variable.
4. Split the data into training and testing sets.
5. Train the Linear Regression model.
6. Evaluate predictions using suitable regression metrics.
7. Integrate the prediction functionality into a Streamlit application.
8. Display the predicted flight price through the application.

## Evaluation Metrics

- **Mean Squared Error (MSE):** Measures the average squared prediction error.
- **Root Mean Squared Error (RMSE):** Measures prediction error in the same units as the target price.
- **R-squared (R²):** Measures how much variation in the target variable is explained by the model.

Include only the metrics actually calculated in the project code.

## Application

The Streamlit interface allows users to enter the required flight details and obtain a predicted ticket price.

## How to Run the Project

1. Clone this repository.
2. Open the project directory.
3. Create and activate a Python virtual environment.
4. Install the required dependencies.
5. Run the Streamlit application.

Example commands:

```bash
git clone <your-repository-url>
cd flight-price-prediction
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application using your actual Streamlit filename:

```bash
streamlit run app.py
```

## Learning Outcomes

- Data preprocessing and exploratory data analysis
- Understanding supervised machine learning
- Implementing Linear Regression
- Evaluating regression models
- Developing an interactive application with Streamlit

## Disclaimer

Predicted prices are estimates based on the available training data and model. They are not guaranteed to match actual airline ticket prices.
