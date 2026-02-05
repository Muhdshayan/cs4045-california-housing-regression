Project README
==============

Course: Deep Learning for Perception CS4045
Author: Muhammad Shayan Memon
Roll No: 22i-0773

Project Files
-------------
- 22i-0773_shayan_A1.ipynb : Complete Jupyter notebook containing all phases of the assignment.
- 22i-0773_dlp_report.pdf : Formal research report documenting experiments and findings.
- README.txt : This file providing overview and instructions.

Project Overview
----------------
Topic: Experimental Linear Regression Analysis of the California Housing Dataset
Instructor: Dr. Ahmad Raza Shahid

This project is organized into five phases:
Phase 1 - Dataset Loading
- Load California Housing dataset from sklearn (no Pandas).
- Display features/targets in NumPy array form.
- Print headers and a table-style view (without DataFrame).

Phase 2 - Exploratory Data Analysis (EDA)
- Descriptive statistics using NumPy (mean, median, min, max, std, etc.).
- Skewness computed for each feature with evidence-based interpretation.
- Histograms and boxplots for distributions.
- Correlation matrix using NumPy + heatmap visualization.
- Discussion on why inter-feature correlations matter.

Phase 3 - Regression Experiments
Part A (Single Feature):
- Linear Regression and SGD Regressor on one predictor.
- Polynomial regression (degree 2 and 3).
- Metrics: MSE, MAE, R2, RMSE.
Part B (Multi-Feature + Engineered):
- Multiple features (e.g., MedInc, HouseAge, AveRooms).
- Engineered square/cubic features.
- Comparison table and overfitting discussion.

Phase 4 - Models Implementation
- 80/20 train-test split.
- Standardization (StandardScaler) with justification.
- Linear Regression and SGD Regressor.
- Train/Test metrics + residual analysis plots.
- Manual comparison table (no Pandas required).

Phase 5 - K-Fold Cross-Validation
- 5-fold CV for Linear Regression and SGD Regressor (scaled).
- Metrics table (MSE, MAE, RMSE, R2).
- Residual plots and stability/performance discussion.

Problem Statement (Detailed)
----------------------------
This project investigates how well linear regression models can predict median house
values in the California Housing dataset. The goal is to conduct systematic experiments
to understand data characteristics (distribution, skewness, correlations) and to compare
model performance under different settings: single-feature vs. multi-feature inputs,
linear vs. polynomial relationships, and train/test split vs. cross-validation. The
dataset contains 20,640 records with eight predictors (Median Income, House Age,
Average Rooms, Average Bedrooms, Population, Average Occupation, Latitude, Longitude)
and one target (Median House Value). The work emphasizes experimental evidence, metric
based evaluation, and interpretation of results.

What I Did (Detailed)
--------------------
1) Dataset Loading (No Pandas)
   - Loaded California Housing data from sklearn.
   - Extracted features (X), target (y), and headers as NumPy arrays.
   - Printed array samples and created a table-style view without DataFrames.

2) Exploratory Data Analysis (NumPy-driven)
   - Computed descriptive statistics for each feature and target (mean, median, min,
     max, std, variance, quartiles).
   - Calculated skewness per feature and interpreted the direction/magnitude with
     evidence.
   - Visualized distributions using histograms and boxplots.
   - Built a NumPy correlation matrix, analyzed feature-target correlations, and
     plotted a heatmap.
   - Discussed why inter-feature correlations matter (multicollinearity, redundancy,
     interpretability, and model stability).

3) Regression Experiments
   Part A: Single-Feature
   - Trained Linear Regression and SGD Regressor on one selected predictor.
   - Added polynomial features (degree 2 and 3) and compared errors.
   - Evaluated using MSE, MAE, R2, and RMSE and discussed fit vs. overfitting.

   Part B: Multi-Feature + Feature Engineering
   - Trained models on multiple predictors (e.g., MedInc, HouseAge, AveRooms).
   - Generated square and cubic feature expansions.
   - Compared Linear Regression and SGD across original vs. engineered features.
   - Summarized results in a comparison table with metric-based conclusions.

4) Model Implementation and Evaluation
   - Performed an 80/20 train-test split.
   - Standardized features using StandardScaler and justified the choice.
   - Trained Linear Regression and SGD Regressor on scaled data.
   - Reported train/test metrics, built comparison tables, and visualized residuals.
   - Interpreted coefficients and compared generalization gaps.

5) K-Fold Cross-Validation
   - Applied 5-fold CV with scaling for both models.
   - Reported mean CV metrics (MSE, MAE, RMSE, R2).
   - Visualized residual distributions across folds.
   - Compared stability and confirmed whether CV aligns with the single split.

Instructions
------------
1. Open the notebook `22i-0773_shayan_A1.ipynb` in VS Code, Google Colab, or any Jupyter environment.
2. Run the notebook cells sequentially from Phase 1 to Phase 5 to reproduce all results.
3. Ensure all required libraries are installed:
   - numpy
   - scikit-learn
   - matplotlib
   - seaborn
   - scipy
4. Refer to the report (`22i-0773_dlp_report.pdf`) for detailed methodology, analysis, and conclusions.
5. Submit the three files (`22i-0773_shayan_A1.ipynb`, `22i-0773_dlp_report.pdf`, and `README.txt`) as the final assignment package.

Notes
-----
- The notebook does not use Pandas; all computations are done with NumPy arrays.
- Recommended Python version: 3.8+.
- Follow academic integrity and assignment guidelines strictly.

Thank you.
