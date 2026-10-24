import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Flight Fare Prediction",
    page_icon="✈️",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("✈️ Flight Fare Prediction & Analysis")

st.write(
    """
    Upload your flight booking dataset, select the flight class,
    choose the train/test split and train a machine learning model
    to predict ticket prices.
    """
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("⚙️ Model Configuration")


# Dataset upload
business_file = st.sidebar.file_uploader(
    "Upload Business Dataset",
    type=["csv"],
    key="business"
)

economy_file = st.sidebar.file_uploader(
    "Upload Economy Dataset",
    type=["csv"],
    key="economy"
)


if business_file is None or economy_file is None:

    st.info(
        "👈 Please upload both Business and Economy datasets."
    )

    st.stop()

# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------
try:

    business_df = pd.read_csv(business.csv)
    economy_df = pd.read_csv(economy.csv)

except Exception as e:

    st.error(f"Unable to read the datasets: {e}")

    st.stop()


# Remove extra spaces from column names

business_df.columns = business_df.columns.str.strip()
economy_df.columns = economy_df.columns.str.strip()


# Add Class column

business_df["Class"] = "Business"
economy_df["Class"] = "Economy"


# Combine both datasets

df = pd.concat(
    [business_df, economy_df],
    ignore_index=True
)

# ---------------------------------------------------------
# DATASET INFORMATION
# ---------------------------------------------------------

st.subheader("📊 Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", f"{df.shape[0]:,}")

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Missing Values", f"{df.isna().sum().sum():,}")


with st.expander("View Dataset"):

    st.dataframe(
        df.head(100),
        use_container_width=True
    )


# ---------------------------------------------------------
# CHECK REQUIRED COLUMNS
# ---------------------------------------------------------

required_columns = [
    "Class",
    "price"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]


if missing_columns:

    st.error(
        f"Required columns are missing: {missing_columns}"
    )

    st.stop()


# ---------------------------------------------------------
# SIDEBAR - CLASS SELECTION
# ---------------------------------------------------------

st.sidebar.subheader("🎫 Flight Class")

classes = ["All"] + sorted(
    df["Class"].dropna().astype(str).unique().tolist()
)

selected_class = st.sidebar.selectbox(
    "Select Class",
    classes
)


# ---------------------------------------------------------
# FILTER CLASS
# ---------------------------------------------------------

filtered_df = df.copy()


if selected_class != "All":

    filtered_df = filtered_df[
        filtered_df["Class"].astype(str) == selected_class
    ]


st.sidebar.write(
    f"Records available: **{len(filtered_df):,}**"
)


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------

st.sidebar.subheader("📚 Train / Test Data")

train_percentage = st.sidebar.slider(
    "Training Data (%)",
    min_value=50,
    max_value=90,
    value=80,
    step=5
)

test_percentage = 100 - train_percentage

st.sidebar.write(
    f"Training: **{train_percentage}%**"
)

st.sidebar.write(
    f"Testing: **{test_percentage}%**"
)


# ---------------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------------

st.sidebar.subheader("🤖 Machine Learning Model")

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "Random Forest",
        "Linear Regression"
    ]
)


# ---------------------------------------------------------
# REMOVE MISSING VALUES
# ---------------------------------------------------------
# ---------------------------------------------------------
# REMOVE MISSING VALUES
# ---------------------------------------------------------

model_df = filtered_df.copy()


# ---------------------------------------------------------
# CLEAN PRICE
# ---------------------------------------------------------

model_df["price"] = (
    model_df["price"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

model_df["price"] = pd.to_numeric(
    model_df["price"],
    errors="coerce"
)

model_df = model_df.dropna(
    subset=["price"]
)

# ---------------------------------------------------------
# REMOVE UNNECESSARY COLUMNS
# ---------------------------------------------------------

# Flight code has many unique values and is usually not
# useful for a simple beginner-level prediction model.

columns_to_remove = [
    "time_taken",
    "ch_code",
    "num_code"
]


# ---------------------------------------------------------
# CREATE FEATURES
# ---------------------------------------------------------

# ---------------------------------------------------------
# CONVERT FLIGHT DURATION TO MINUTES
# ---------------------------------------------------------

def convert_duration_to_minutes(value):

    try:

        value = str(value).strip()

        hours = 0
        minutes = 0

        if "h" in value:

            hours = int(
                value.split("h")[0].strip()
            )

        if "m" in value:

            if "h" in value:

                minutes_part = (
                    value.split("h")[1]
                    .replace("m", "")
                    .strip()
                )

            else:

                minutes_part = (
                    value.replace("m", "")
                    .strip()
                )

            if minutes_part:

                minutes = int(minutes_part)

        return hours * 60 + minutes

    except:

        return np.nan


model_df["duration_minutes"] = (
    model_df["time_taken"]
    .apply(convert_duration_to_minutes)
)


# ---------------------------------------------------------
# CREATE X AND y
# ---------------------------------------------------------

X = model_df.drop(
    columns=["price", "price"] + columns_to_remove,
    errors="ignore"
)

y = model_df["price"]


# ---------------------------------------------------------
# IDENTIFY NUMERIC / CATEGORICAL FEATURES
# ---------------------------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )

    ],
    remainder="passthrough"
)


# ---------------------------------------------------------
# SELECT MODEL
# ---------------------------------------------------------

if model_name == "Random Forest":

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

else:

    model = LinearRegression()


# ---------------------------------------------------------
# CREATE PIPELINE
# ---------------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ---------------------------------------------------------
# TRAIN MODEL BUTTON
# ---------------------------------------------------------

train_button = st.sidebar.button(
    "🚀 Train Model",
    type="primary",
    use_container_width=True
)


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

if train_button:

    if len(model_df) < 10:

        st.error(
            "Not enough data to train the model."
        )

        st.stop()


    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_percentage / 100,
        random_state=42
    )


    # Train model
    with st.spinner("Training machine learning model..."):

        pipeline.fit(
            X_train,
            y_train
        )


    # Predictions
    y_pred = pipeline.predict(
        X_test
    )


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )


    # -----------------------------------------------------
    # TOTAL AMOUNT
    # -----------------------------------------------------

    actual_total = y_test.sum()

    predicted_total = y_pred.sum()

    total_difference = predicted_total - actual_total


    # -----------------------------------------------------
    # SAVE RESULTS IN SESSION
    # -----------------------------------------------------

    st.session_state["trained"] = True
    st.session_state["model"] = pipeline
    st.session_state["y_test"] = y_test
    st.session_state["y_pred"] = y_pred

    st.session_state["mse"] = mse
    st.session_state["rmse"] = rmse
    st.session_state["r2"] = r2

    st.session_state["actual_total"] = actual_total
    st.session_state["predicted_total"] = predicted_total
    st.session_state["difference"] = total_difference

    st.success("✅ Model trained successfully!")


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

if st.session_state.get("trained", False):

    st.divider()

    st.subheader("📈 Model Performance")


    # -----------------------------------------------------
    # METRIC CARDS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "MSE",
            f"{st.session_state['mse']:,.2f}"
        )


    with col2:

        st.metric(
            "RMSE",
            f"{st.session_state['rmse']:,.2f}"
        )


    with col3:

        st.metric(
            "R² Score",
            f"{st.session_state['r2']:.4f}"
        )


    with col4:

        st.metric(
            "Test Records",
            f"{len(st.session_state['y_test']):,}"
        )


    # -----------------------------------------------------
    # TOTAL AMOUNT
    # -----------------------------------------------------

    st.divider()

    st.subheader("💰 Total Ticket Amount")


    amount_col1, amount_col2, amount_col3 = st.columns(3)


    with amount_col1:

        st.metric(
            "Actual Total",
            f"₹ {st.session_state['actual_total']:,.2f}"
        )


    with amount_col2:

        st.metric(
            "Predicted Total",
            f"₹ {st.session_state['predicted_total']:,.2f}"
        )


    with amount_col3:

        st.metric(
            "Total Error",
            f"₹ {st.session_state['difference']:,.2f}"
        )


    # -----------------------------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "🎯 Actual Price vs Predicted Price"
    )


    y_test = st.session_state["y_test"]
    y_pred = st.session_state["y_pred"]


    fig1, ax1 = plt.subplots(
        figsize=(10, 5)
    )


    ax1.scatter(
        y_test,
        y_pred,
        alpha=0.5
    )


    # Perfect prediction line
    min_value = min(
        y_test.min(),
        y_pred.min()
    )

    max_value = max(
        y_test.max(),
        y_pred.max()
    )


    ax1.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )


    ax1.set_xlabel(
        "Actual Price"
    )

    ax1.set_ylabel(
        "Predicted Price"
    )

    ax1.set_title(
        "Actual vs Predicted Flight Price"
    )


    st.pyplot(fig1)


    # -----------------------------------------------------
    # ERROR CALCULATION
    # -----------------------------------------------------

    errors = y_test.values - y_pred


    # -----------------------------------------------------
    # ERROR DISTRIBUTION
    # -----------------------------------------------------

    st.subheader(
        "📉 Prediction Error Distribution"
    )


    fig2, ax2 = plt.subplots(
        figsize=(10, 5)
    )


    ax2.hist(
        errors,
        bins=40
    )


    ax2.axvline(
        0,
        linestyle="--"
    )


    ax2.set_xlabel(
        "Prediction Error"
    )

    ax2.set_ylabel(
        "Number of Flights"
    )

    ax2.set_title(
        "Distribution of Prediction Errors"
    )


    st.pyplot(fig2)


    # -----------------------------------------------------
    # ERROR BY OBSERVATION
    # -----------------------------------------------------

    st.subheader(
        "📊 Error for Test Data"
    )


    error_df = pd.DataFrame({

        "Actual Price": y_test.values,

        "Predicted Price": y_pred,

        "Error": errors,

        "Absolute Error": np.abs(errors)

    })


    st.dataframe(
        error_df.head(100),
        use_container_width=True
    )


    # -----------------------------------------------------
    # ERROR VISUALIZATION
    # -----------------------------------------------------

    fig3, ax3 = plt.subplots(
        figsize=(12, 5)
    )


    ax3.plot(
        np.arange(len(errors[:200])),
        errors[:200]
    )


    ax3.axhline(
        0,
        linestyle="--"
    )


    ax3.set_xlabel(
        "Test Data Index"
    )

    ax3.set_ylabel(
        "Prediction Error"
    )

    ax3.set_title(
        "Prediction Error for Test Data"
    )


    st.pyplot(fig3)


    # -----------------------------------------------------
    # MODEL INTERPRETATION
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "🧠 Model Interpretation"
    )


    if r2 >= 0.90:

        st.success(
            "Excellent model performance. "
            "The model explains most of the variation in ticket prices."
        )

    elif r2 >= 0.70:

        st.info(
            "Good model performance. "
            "There is still room for improvement."
        )

    elif r2 >= 0.50:

        st.warning(
            "Moderate model performance. "
            "Feature engineering or a different model may improve the result."
        )

    else:

        st.error(
            "The model performance is relatively low. "
            "Consider improving preprocessing, feature engineering, "
            "or trying another algorithm."
        )


    st.write(
        f"""
        **MSE:** {mse:,.2f}

        **RMSE:** ₹ {rmse:,.2f}

        **R² Score:** {r2:.4f}

        **Actual Total:** ₹ {actual_total:,.2f}

        **Predicted Total:** ₹ {predicted_total:,.2f}
        """
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "✈️ Flight Fare Prediction | Built with Python, "
    "Pandas, Scikit-learn and Streamlit"
)
