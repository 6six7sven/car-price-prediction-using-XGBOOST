
# Car Price Prediction Model

This project explores the development of a machine learning model to predict the selling price of used cars based on various features.

**Key Functionalities:**

* **Data Preprocessing:**
  * Imports necessary libraries (`numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `warnings`).
  * Reads the car data from a CSV file (`Car_extended.csv`).
  * Handles missing values by identifying rows with missing data (`mileage`, `engine`, `seats`) and dropping them (`dropna`).
  * Cleans and transforms categorical features:
    * Converts `max_power` values to numerical by splitting on whitespace and handling potential non-string values.
    * Removes units from `engine` by replacing "CC".
    * Converts `mileage` values to float by splitting on whitespace and handling potential non-string values.
    * Drops irrelevant columns (`torque`, `name`).
    * Calculates car age (`age`) by subtracting the year from 2021 and drops the original `year` column.
    * One-hot encodes categorical columns (`owner`, `fuel`, `seller_type`, `transmission`) using Pandas' `get_dummies` to create numerical representations suitable for machine learning models.
    * Handles remaining missing values (`owner`) by dropping rows with missing values (`dropna`).
* **Exploratory Data Analysis (EDA):**
  * Analyzes data distribution and relationships between features:
    * Displays the first few rows (`data.head()`).
    * Counts non-null values in each column (`data.count()`).
    * Counts missing values in each column (`data.isnull().sum()`).
    * Creates visualizations using Seaborn to explore relationships between features and the target variable (`selling_price`):
      * Bar plots for categorical features (`owner`, `transmission`, `seats`, `fuel`, `seller_type`).
      * Relationship between car age (`age`) and `selling_price`.
      * Scatter plots for numerical features (`km_driven`, `mileage`, `engine`, `max_power`) and `selling_price`.
* **Feature Engineering:**
  * Creates dummy variables for categorical features using Pandas' `get_dummies`.
* **Model Building and Evaluation:**
  * Splits the data into training and testing sets (`train_test_split`) for model training and evaluation.
  * Applies MinMax scaling (`MinMaxScaler`) to normalize numerical features in both training and testing sets.
  * Trains a linear regression model (`LinearRegression`).
  * Evaluates the model's performance using:
    * Root Mean Squared Error (RMSE)
    * Mean Absolute Error (MAE)
    * R-squared score (`r2_score`)
  * Explores alternative models like XGBoost (`XGBRegressor`) to potentially improve performance.

**Future Enhancements:**

* **Feature Selection:** Employ techniques like correlation analysis or feature importance scores to identify the most impactful features for model performance.
* **Model Tuning:** Experiment with hyperparameter tuning for the chosen model(s) to potentially improve accuracy.
* **Cross-Validation:** Implement cross-validation techniques to obtain a more robust estimate of model performance on unseen data.
* **Deployment:** Consider deploying the model as a web application or API for real-world car price prediction.

**Running the Script:**

1. Ensure you have the required libraries installed (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`).
2. Replace the file path in `data = pd.read_csv("F:\Code-Space\carPrice\Car-Price-Prediction\Car_extended.csv")` with the actual location of your CSV file.
3. Execute the script line by line or as a whole.

**Disclaimer:**

The model's accuracy may vary depending on the quality and completeness of the data used.
