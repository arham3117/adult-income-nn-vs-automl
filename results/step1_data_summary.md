# Step 1 - Dataset and Preprocessing Summary

Random seed: **42**  |  Split: **60/20/20 stratified**

## Split sizes and class balance

| Split | Rows | <=50K (0) | >50K (1) | % >50K |
|---|---|---|---|---|
| train | 29,274 | 22,265 | 7,009 | 23.943% |
| val | 9,758 | 7,422 | 2,336 | 23.939% |
| test | 9,758 | 7,422 | 2,336 | 23.939% |
| total | 48,790 | 37,109 | 11,681 | 23.941% |

Raw features used: **12** (5 numeric, 7 categorical)
Features after one-hot encoding: **91**

## Missing values in the raw data

| Column | Missing rows |
|---|---|
| workclass | 2,799 |
| occupation | 2,809 |
| native-country | 857 |

## Preprocessing decisions

1. Merged adult.data and adult.test into one pool, then re-split ourselves so that train/val/test come from an identical distribution and the split is reproducible.
2. Stripped whitespace on all string columns and removed the trailing '.' that adult.test appends to its labels.
3. Mapped the target to binary: '<=50K' -> 0, '>50K' -> 1.
4. Dropped ['fnlwgt', 'education']: fnlwgt is a census sampling weight (a survey design artefact, not a person-level predictor); education is a string duplicate of the ordinal education-num.
5. Dropped 52 exact duplicate rows to avoid the same record appearing in both train and test.
6. '?' tokens were read as NaN and imputed with an explicit 'Missing' category rather than the mode, because missingness in workclass/occupation is informative (it co-occurs with 'Never-worked'/retired respondents).
7. Numeric features: median imputation (robust to the heavy right skew of capital-gain) followed by StandardScaler, which matters for gradient descent.
8. Categorical features: OneHotEncoder(handle_unknown='ignore') so an unseen category at inference yields an all-zero block instead of an error.
9. The ColumnTransformer is fit on the training split only; validation and test are transformed with those fitted statistics to prevent data leakage.
10. Kept capital-gain/capital-loss despite ~91% zeros: they are among the strongest signals for the >50K class.
