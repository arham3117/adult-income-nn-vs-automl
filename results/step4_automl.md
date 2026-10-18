# Step 4 - AutoML (FLAML) Results

Framework: **FLAML 2.3.6** | metric: **roc_auc** | time budget: **300s** | actual search: **300.1s** | trials: **13** | seed: **42**

Validation is the *same* Step-1 validation split (`eval_method='holdout'`, `X_val`/`y_val` passed explicitly). `retrain_full=False` so FLAML never trains on validation rows.

## Leaderboard (best configuration per estimator)

| Rank | Estimator | Val ROC-AUC | Val acc | Val F1 | Test ROC-AUC | Test acc | Test F1 |
|---|---|---|---|---|---|---|---|
| 1 | xgb_limitdepth | 0.9328 | 0.8768 | 0.7209 | 0.9266 | 0.8711 | 0.7121 |
| 2 | lgbm | 0.9306 | 0.8747 | 0.7205 | 0.9245 | 0.8699 | 0.7070 |
| 3 | xgboost | 0.9299 | 0.8756 | 0.7097 | 0.9245 | 0.8696 | 0.6945 |
| 4 | rf | 0.9214 | 0.8692 | 0.6922 | 0.9155 | 0.8630 | 0.6771 |
| 5 | extra_tree | 0.9193 | 0.8684 | 0.6950 | 0.9128 | 0.8607 | 0.6745 |
| 6 | lrl1 | 0.4851 | 0.7942 | 0.4059 | 0.4881 | 0.7891 | 0.3929 |

## Selected model: `xgb_limitdepth`

Hyperparameters chosen by FLAML:

| Hyperparameter | Value |
|---|---|
| n_estimators | 2770 |
| max_depth | 7 |
| min_child_weight | 0.09334064712939191 |
| learning_rate | 0.006200219332923492 |
| subsample | 0.9960500864194881 |
| colsample_bylevel | 0.8155116309985395 |
| colsample_bytree | 0.4012180435147734 |
| reg_alpha | 0.008095607070800785 |
| reg_lambda | 0.0665179393932735 |

## Selected model test metrics

| Metric | Value |
|---|---|
| accuracy | 0.8711 |
| precision | 0.7650 |
| recall | 0.6661 |
| f1 | 0.7121 |
| roc_auc | 0.9266 |

| | Predicted <=50K | Predicted >50K |
|---|---|---|
| **Actual <=50K** | 6,944 | 478 |
| **Actual >50K** | 780 | 1,556 |

