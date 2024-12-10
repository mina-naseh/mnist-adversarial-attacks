# mnist-adversarial-attacks

## Data Preparation
- Loaded the MNIST dataset using `fetch_openml`.
- Split the data into training, validation, and test sets.
- Preprocessed the data:
  - Reshaped each image to 28×28.
  - Normalized pixel values to the range [0, 1].
- Converted the data into PyTorch tensors for use with neural networks.

## Model Definition
- Implemented a convolutional neural network (`Net`) with:
  - Two convolutional layers.
  - Dropout for regularization.
  - Two fully connected layers for classification.
- Model outputs probabilities for 10 classes (digits 0–9).

## Training
- Implemented training logic using `train_loop` and `train_model`:
  - `train_loop`: Handles one epoch of training.
  - `train_model`: Orchestrates the full training process with validation after each epoch.
- Used:
  - SGD optimizer with a learning rate of 0.001 and momentum of 0.9.
  - Cross-Entropy Loss for multi-class classification.
  - Batch size of 64 for training.
- Trained the model for 10 epochs, monitoring both training and validation losses.

## Validation
- Evaluated the model after each epoch on a separate validation set.
- Logged:
  - Average validation loss.
  - Validation accuracy.

## Clean Accuracy Evaluation
- Evaluated the model's performance on the unseen test set.
- Calculated clean accuracy by comparing predictions against ground truth labels.
- **Result**: The model achieved a clean accuracy of **90.96%%**.

## Visualization
- Implemented a confusion matrix to analyze class-wise performance.
- Generated and plotted:
  - Confusion matrix in `output` folder.
  - Classification report showing precision, recall, and F1-score for each digit class:
  ```
                    precision    recall  f1-score   support

                0       0.93      0.97      0.95      1381
                1       0.94      0.97      0.96      1575
                2       0.91      0.91      0.91      1398
                3       0.90      0.90      0.90      1428
                4       0.91      0.89      0.90      1365
                5       0.92      0.83      0.87      1263
                6       0.92      0.95      0.94      1375
                7       0.89      0.94      0.92      1459
                8       0.90      0.85      0.88      1365
                9       0.87      0.86      0.86      1391

         accuracy                           0.91     14000
        macro avg       0.91      0.91      0.91     14000
     weighted avg       0.91      0.91      0.91     14000 
    ```