# House-price-prediction-regression-analysis-ml
End-to-end housing price prediction and regression analysis using Python and scikit-learn, including EDA, data preprocessing, feature analysis, model evaluation, and residual analysis.

## Overview

This project uses **Linear Regression** to predict house prices based on different property characteristics.

The main goal of this project is not only to build a prediction model, but also to understand the complete workflow of a supervised machine learning regression problem — from exploratory data analysis and feature selection to model evaluation and interpretation.

The project uses a housing dataset containing information about house size, quality, construction year, number of bedrooms and bathrooms, lot size, floors, and other property characteristics.

---

## Objectives

* Explore and understand the housing dataset.
* Identify relationships between house features and price.
* Use scatter plots and correlation analysis to investigate relationships.
* Build a simple Linear Regression model using a single feature.
* Gradually add multiple features and compare model performance.
* Evaluate models using **Mean Absolute Error (MAE)** and **R² Score**.
* Interpret Linear Regression coefficients.
* Investigate correlations and multicollinearity between predictors.
* Understand how feature selection affects model performance.

---

## Dataset

The dataset contains **21,613 house records** and 21 columns. The target variable is `price`, while the remaining columns describe different characteristics of each property.

| Feature         | Description                                                                                                           |
| --------------- | --------------------------------------------------------------------------------------------------------------------- |
| `id`            | Unique identifier assigned to each house record.                                                                      |
| `date`          | Date on which the house sale was recorded.                                                                            |
| `price`         | Sale price of the house. **Target variable** for prediction.                                                          |
| `bedrooms`      | Number of bedrooms in the house.                                                                                      |
| `bathrooms`     | Number of bathrooms in the house.                                                                                     |
| `sqft_living`   | Interior living area of the house in square feet.                                                                     |
| `sqft_lot`      | Total area of the property/lot in square feet.                                                                        |
| `floors`        | Number of floors in the house.                                                                                        |
| `waterfront`    | Indicates whether the property has a waterfront view/access (`0` = No, `1` = Yes).                                    |
| `view`          | Indicates the quality of the property's view, represented by an ordinal score.                                        |
| `condition`     | Overall condition of the house, represented by a rating/score.                                                        |
| `grade`         | Overall construction and design quality of the house, represented by a rating/score.                                  |
| `sqft_above`    | Living area located above ground level, measured in square feet.                                                      |
| `sqft_basement` | Basement area of the house in square feet.                                                                            |
| `yr_built`      | Year in which the house was originally built.                                                                         |
| `yr_renovated`  | Year in which the house was last renovated. A value of `0` generally indicates that the house has not been renovated. |
| `zipcode`       | ZIP code identifying the geographical location of the property.                                                       |
| `lat`           | Latitude coordinate of the property's location.                                                                       |
| `long`          | Longitude coordinate of the property's location.                                                                      |
| `sqft_living15` | Average living area of the 15 nearest houses, measured in square feet.                                                |
| `sqft_lot15`    | Average lot size of the 15 nearest houses, measured in square feet.                                                   |

---

## Exploratory Data Analysis

The dataset was first examined to understand its structure and quality.

The analysis included:

* Checking data types
* Checking for missing values
* Checking for duplicate records
* Examining feature distributions
* Identifying skewed variables
* Creating scatter plots between important features and house price

### Key observations

#### Living Area vs Price

` sqft_living` showed a strong positive relationship with price. In general, houses with larger living areas tend to have higher prices.

However, the spread of prices also increases for larger houses, suggesting that living area alone cannot explain all differences in price.

#### Lot Size vs Price

`sqft_lot` did not show a clear linear relationship with price.

The feature is also highly skewed, with most observations concentrated at relatively smaller lot sizes and a smaller number of houses having extremely large lots.

#### Grade vs Price

`grade` showed a strong positive relationship with price. Houses with higher grades generally had higher prices.

#### Year Built vs Price

Newer houses generally showed higher prices, although houses built in the same year could still have substantially different prices.

#### Bedrooms, Bathrooms, Floors and Condition

These variables generally showed positive associations with price, although the relationships were not as straightforward as `sqft_living` and `grade`.

---

## Modeling Approach

### Step 1 — Single Feature Model

The first model used only:

```text
sqft_living → price
```

This was used as a baseline to understand how much predictive information a single important feature could provide.

### Step 2 — Adding Grade

`grade` was added because the exploratory analysis showed a strong relationship between house quality and price.

### Step 3 — Adding Bedrooms and Bathrooms

The number of bedrooms and bathrooms was added to provide additional information about the characteristics of the property.

### Step 4 — Adding Year Built, Condition and Floors

Additional structural and quality-related features were added.

### Step 5 — Additional Features

`sqft_lot`, `sqft_above`, and `yr_renovated` were also tested.

The effect of `zipcode` was investigated separately. Adding it directly as a numerical feature reduced test-set R². Since ZIP codes represent categories/locations rather than continuous numerical quantities, treating them directly as numbers is not necessarily appropriate. Further location-based feature engineering can be explored in future versions.

---

## Model Performance

The models were evaluated using:

### Mean Absolute Error (MAE)

MAE represents the average absolute difference between the actual and predicted house prices.

For example, an MAE of approximately `$140,000` means that the model's predictions are, on average, about `$140,000` away from the actual prices.

Lower MAE is better.

### R² Score

R² measures how much of the variation in the target variable is explained by the model.

An R² value of:

* `1` → perfect fit
* `0` → no improvement over predicting the mean
* `< 0` → worse than the mean baseline

R² should not be interpreted as prediction accuracy.

---

## Model Comparison

| Model   | Features                                   |             MAE |         R² |
| ------- | ------------------------------------------ | --------------: | ---------: |
| Model 1 | `sqft_living`                              |     $170,780.91 |     0.4792 |
| Model 2 | `sqft_living`, `grade`                     |     $161,048.44 |     0.5372 |
| Model 3 | + `bathrooms`, `bedrooms`                  |     $158,262.01 |     0.5463 |
| Model 4 | + `yr_built`, `condition`, `floors`        |     $140,248.01 |     0.6231 |
| Model 5 | + `sqft_lot`, `sqft_above`, `yr_renovated` | **$140,091.73** | **0.6246** |
| Model 6 | Model 5 without `sqft_above`               |     $140,216.08 |     0.6242 |

### Current Best Model

The best-performing model tested so far is **Model 5**.

```text
MAE = $140,091.73
R²  = 0.6246
```

The model explains approximately **62.46% of the variation in house prices on the test set**.

The relatively small difference between Models 5 and 6 also shows that `sqft_above` provides only a small additional improvement when the other features are already present.

---

## Coefficient Analysis

The coefficients learned by the current model were examined to understand how Linear Regression uses the features to generate predictions.

Some coefficients were positive while others were negative.

For example:

* `sqft_living` → positive
* `grade` → positive
* `bathrooms` → positive
* `condition` → positive
* `floors` → positive
* `bedrooms` → negative
* `yr_built` → negative
* `sqft_above` → slightly negative

The negative coefficients do not necessarily mean that these variables have a negative relationship with house price.

This is because multiple linear regression estimates the effect of each feature **while holding the other features constant**.

---

## Correlation and Multicollinearity

A correlation matrix was used to investigate relationships between predictor variables.

Some strong correlations were observed:

```text
sqft_living ↔ sqft_above    = 0.877
sqft_living ↔ grade         = 0.763
sqft_living ↔ bathrooms     = 0.755
sqft_above  ↔ grade         = 0.756
```

These relationships indicate that several predictors contain overlapping information.

This is known as **multicollinearity**.

Multicollinearity can make individual regression coefficients harder to interpret because multiple features are explaining similar aspects of the data.

However, a highly correlated feature should not automatically be removed when the primary goal is prediction. Model performance should also be considered.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

## Project Structure

```text
house-price-linear-regression/
│
├── data/
│   └── housing.csv
│
├── notebooks/
│   └── house_price_prediction.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## 🚀 Future Improvements

The current project focuses on understanding Linear Regression. Several improvements can be explored:

* Perform cross-validation for more reliable evaluation.
* Investigate location-related features such as latitude and longitude.
* Properly encode categorical location features such as ZIP code.
* Apply transformations to highly skewed variables such as `sqft_lot`.
* Perform residual analysis.
* Investigate multicollinearity using VIF.
* Compare Linear Regression with regularized models such as Ridge and Lasso.
* Compare the results with tree-based regression models.
* Perform more systematic feature engineering.
* Build a simple interface for making house-price predictions.

---

## 📌 Conclusion

This project demonstrates the complete basic workflow of a Linear Regression problem:

```text
Data Understanding
       ↓
Exploratory Data Analysis
       ↓
Feature Selection
       ↓
Train/Test Split
       ↓
Linear Regression
       ↓
MAE + R² Evaluation
       ↓
Coefficient Analysis
       ↓
Correlation & Multicollinearity
```

The experiments show that house price can be predicted using property characteristics, with the current best model achieving an **R² of 0.6246** and an **MAE of approximately $140,092** on the test set.

The project will be further improved through feature engineering, cross-validation, location analysis, and residual analysis.
