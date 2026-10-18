# Step 3 - Manual Tuning (12 configurations)

Sorted by validation ROC-AUC. All runs use the identical Step-1 split and seed 42.

| config | hidden_sizes | lr | batch_size | dropout | weight_decay | batch_norm | early_stopping | params | epochs_run | train_time_s | val_loss | val_accuracy | val_f1 | val_roc_auc | train_val_acc_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C3-wide | 512-256 | 0.001 | 256 | 0.3 | 0.0 | False | True | 178689 | 15 | 2.793 | 0.2999 | 0.8626 | 0.6803 | 0.9182 | 0.008 |
| C11-dropout-heavy | 256-128 | 0.001 | 256 | 0.5 | 0.0 | False | True | 56577 | 23 | 2.671 | 0.3005 | 0.8634 | 0.6883 | 0.918 | 0.0077 |
| C12-batchnorm-wd | 256-128 | 0.001 | 256 | 0.2 | 0.0001 | True | True | 57345 | 15 | 2.292 | 0.2998 | 0.8613 | 0.688 | 0.9178 | 0.0039 |
| C1-baseline | 128-64 | 0.001 | 256 | 0.3 | 0.0 | False | True | 20097 | 23 | 1.97 | 0.3013 | 0.8613 | 0.6863 | 0.9174 | 0.0045 |
| C9-batch-large | 128-64 | 0.001 | 1024 | 0.3 | 0.0 | False | True | 20097 | 28 | 2.117 | 0.3006 | 0.8613 | 0.6937 | 0.9174 | 0.0029 |
| C5-deep-4layer | 256-128-64-32 | 0.001 | 256 | 0.3 | 0.0 | False | True | 66817 | 16 | 2.162 | 0.301 | 0.8617 | 0.684 | 0.9172 | 0.0037 |
| C7-lr-low | 128-64 | 0.0003 | 256 | 0.3 | 0.0 | False | True | 20097 | 31 | 2.755 | 0.301 | 0.8607 | 0.684 | 0.9172 | 0.0039 |
| C8-batch-small | 128-64 | 0.001 | 64 | 0.3 | 0.0 | False | True | 20097 | 15 | 2.545 | 0.3016 | 0.8622 | 0.678 | 0.9172 | 0.0031 |
| C4-deep-3layer | 256-128-64 | 0.001 | 256 | 0.3 | 0.0 | False | True | 64769 | 15 | 1.974 | 0.3012 | 0.8606 | 0.6746 | 0.917 | 0.0059 |
| C6-lr-high | 128-64 | 0.01 | 256 | 0.3 | 0.0 | False | True | 20097 | 15 | 1.268 | 0.3039 | 0.861 | 0.6615 | 0.9169 | 0.0057 |
| C2-shallow-narrow | 32-16 | 0.001 | 256 | 0.3 | 0.0 | False | True | 3489 | 35 | 2.347 | 0.3027 | 0.8609 | 0.6883 | 0.9168 | 0.0017 |
| C10-no-reg | 256-128 | 0.001 | 256 | 0.0 | 0.0 | False | False | 56577 | 60 | 5.196 | 0.301 | 0.8483 | 0.6693 | 0.8933 | 0.0525 |

## Seed-only noise floor

Configuration C1 retrained under three different seeds on the same split:

| Seed | val ROC-AUC | val accuracy | val F1 |
|---|---|---|---|
| 0 | 0.9178 | 0.8627 | 0.6907 |
| 7 | 0.9172 | 0.8600 | 0.6888 |
| 123 | 0.9176 | 0.8607 | 0.6858 |

Seed-only spread: ROC-AUC 0.0006, accuracy 0.0027, F1 0.0049. A factor only counts as real if its spread exceeds this band.

## Which factor mattered most

| Factor | ROC-AUC spread | Accuracy spread | F1 spread | Train-time spread (s) | > seed noise (AUC) |
|---|---|---|---|---|---|
| learning rate (C1 / C6 / C7) | 0.0005 | 0.0006 | 0.0248 | 1.5 | no |
| depth (C1 / C4 / C5) | 0.0004 | 0.0011 | 0.0117 | 0.2 | no |
| width (C2 / C1 / C3) | 0.0014 | 0.0017 | 0.0080 | 0.8 | yes |
| batch size (C8 / C1 / C9) | 0.0002 | 0.0009 | 0.0157 | 0.6 | no |
| regularisation (C10 / C11 / C12) | 0.0247 | 0.0151 | 0.0190 | 2.9 | yes |

**Largest raw ROC-AUC spread: regularisation (C10 / C11 / C12)**

## Overfitting evidence (final-epoch weights)

| Config | final train acc | final val acc | final val loss | min val loss | val-loss rise |
|---|---|---|---|---|---|
| C1-baseline | 0.8650 | 0.8605 | 0.3072 | 0.3013 | +0.0059 |
| C10-no-reg | 0.9008 | 0.8483 | 0.4310 | 0.3010 | +0.1299 |
| C11-dropout-heavy | 0.8672 | 0.8595 | 0.3061 | 0.3005 | +0.0057 |
| C12-batchnorm-wd | 0.8642 | 0.8603 | 0.3019 | 0.2998 | +0.0021 |

