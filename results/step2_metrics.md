# Step 2 - Baseline Deep Feed-Forward Network

## Architecture

| Layer | Type | Output units | Activation | Parameters |
|---|---|---|---|---|
| Input | - | 91 | - | 0 |
| Hidden 1 (Linear) | Linear(91, 128) | 128 | ReLU | 11,776 |
|   Dropout(p=0.3) | Dropout | 128 | - | 0 |
| Hidden 2 (Linear) | Linear(128, 64) | 64 | ReLU | 8,256 |
|   Dropout(p=0.3) | Dropout | 64 | - | 0 |
| Output (Linear) | Linear(64, 1) | 1 | Sigmoid (via BCEWithLogitsLoss) | 65 |

**Total trainable parameters: 20,097**

Loss: BCEWithLogitsLoss | Optimizer: Adam(lr=0.001) | Batch size: 256 | Max epochs: 100 | Early stopping patience: 10

Trained 23 epochs in 2.0s; best validation loss at epoch 13.

## Test-set performance

| Metric | Value |
|---|---|
| accuracy | 0.8541 |
| precision | 0.7310 |
| recall | 0.6177 |
| f1 | 0.6696 |
| roc_auc | 0.9104 |

## Confusion matrix (test)

| | Predicted <=50K | Predicted >50K |
|---|---|---|
| **Actual <=50K** | 6,891 (TN) | 531 (FP) |
| **Actual >50K** | 893 (FN) | 1,443 (TP) |

