
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

def prepare_data(file_path, test_size=0.2, random_state=42, n_splits=5):
    """
    Loads data, splits it into training and test sets, and prepares for stratified k-fold cross-validation.

    Args:
        file_path (str): Path to the dataset (e.g., CSV file).
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.
        n_splits (int): Number of folds in the cross-validator.

    Returns:
        tuple: X_train, X_test, y_train, y_test, skf (StratifiedKFold object)
    """
    df = pd.read_csv(file_path)

    # Assuming the target variable is named 'cardio'
    X = df.drop('cardio', axis=1)
    y = df['cardio']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print(f"Data loaded from: {file_path}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Stratified K-Fold with {n_splits} splits initialized.")

    return X_train, X_test, y_train, y_test, skf

if __name__ == '__main__':
    # Example usage:
    file_path = 'cardio_engineered.csv'
    X_train, X_test, y_train, y_test, skf = prepare_data(file_path)

    print("\nIterating through Stratified K-Fold splits (example):")
    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        print(f"Fold {fold+1}: Train samples: {len(X_train_fold)}, Validation samples: {len(X_val_fold)}")


