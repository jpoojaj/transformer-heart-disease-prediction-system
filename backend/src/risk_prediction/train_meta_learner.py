
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import optuna

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim_1=64, hidden_dim_2=32, dropout_rate=0.3, num_classes=2):
        super(MLPClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_dim_2, num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.dropout2(self.relu2(self.layer2(x)))
        x = self.output_layer(x)
        return x

def train_and_evaluate_mlp(X_train_meta, y_train_meta, X_val_meta, y_val_meta, X_test, y_test, params, device):
    train_ds = TensorDataset(torch.tensor(X_train_meta, dtype=torch.float32), torch.tensor(y_train_meta, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val_meta, dtype=torch.float32), torch.tensor(y_val_meta, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=params.get('batch_size', 64), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params.get('batch_size', 64), shuffle=False)

    model = MLPClassifier(
        input_dim=X_train_meta.shape[1],
        hidden_dim_1=params.get('hidden_dim_1', 64),
        hidden_dim_2=params.get('hidden_dim_2', 32),
        dropout_rate=params.get('dropout_rate', 0.3)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params.get('lr', 0.001))
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_without_improve = 0
    best_model_state = None

    for epoch in range(params.get('epochs', 100)):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for xb_val, yb_val in val_loader:
                xb_val = xb_val.to(device)
                outputs_val = model(xb_val)
                loss_val = criterion(outputs_val, yb_val.to(device))
                val_loss += loss_val.item()
                probs_val = torch.softmax(outputs_val, dim=1)[:, 1].cpu().numpy()
                all_val_preds.extend(probs_val)
                all_val_labels.extend(yb_val.cpu().numpy())

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= params.get('patience', 10):
                break
    
    if best_model_state: # Load best model found during training
        model.load_state_dict(best_model_state)
    else: # If no improvement was ever made, use the last state
        best_model_state = model.state_dict()
        model.load_state_dict(best_model_state)

    # Evaluate on the test set if provided
    if X_test is not None and y_test is not None:
        test_ds = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.long))
        test_loader = DataLoader(test_ds, batch_size=params.get('batch_size', 64), shuffle=False)
        model.eval()
        all_test_preds = []
        all_test_labels = []
        with torch.no_grad():
            for xb_test, yb_test in test_loader:
                xb_test = xb_test.to(device)
                outputs_test = model(xb_test)
                probs_test = torch.softmax(outputs_test, dim=1)[:, 1].cpu().numpy()
                all_test_preds.extend(probs_test)
                all_test_labels.extend(yb_test.cpu().numpy())
        test_auc = roc_auc_score(all_test_labels, all_test_preds)
    else:
        test_auc = -1.0 # Indicate no test evaluation

    return best_val_loss, test_auc, model.cpu()

def objective(trial, X_meta_train, y_full, X_test, y_test):
    # Hyperparameters to tune
    hidden_dim_1 = trial.suggest_categorical('hidden_dim_1', [32, 64, 128])
    hidden_dim_2 = trial.suggest_categorical('hidden_dim_2', [16, 32, 64])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = 50 # Keep epochs fixed for tuning, rely on early stopping
    patience = 7 # Early stopping patience

    params = {
        'hidden_dim_1': hidden_dim_1,
        'hidden_dim_2': hidden_dim_2,
        'dropout_rate': dropout_rate,
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'patience': patience,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split meta-training data into training and validation sets for early stopping
    X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(
        X_meta_train, y_full.values, test_size=0.2, random_state=42, stratify=y_full.values
    )
    
    val_loss, _, _ = train_and_evaluate_mlp(X_train_meta, y_train_meta, X_val_meta, y_val_meta, None, None, params, device)

    return val_loss

def tune_and_train_meta_learner(X_meta_train, y_full, X_test, y_test, n_trials=50):
    print(f"Starting Optuna hyperparameter tuning for Meta-Learner with {n_trials} trials...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_meta_train, y_full, X_test, y_test), n_trials=n_trials)

    print("\nNumber of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    best_params = trial.params
    best_params['epochs'] = 100 # Train longer with best params
    best_params['patience'] = 15 # More patience for final training

    print("\nTraining final Meta-Learner with best hyperparameters...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use the full meta-training set to train the final model
    final_val_loss, final_test_auc, final_model = train_and_evaluate_mlp(X_meta_train, y_full.values, X_meta_train, y_full.values, X_test, y_test, best_params, device) # Passing X_meta_train for both train and val to use all data for final training
    
    print(f"Final Meta-Learner (tuned) Test AUC: {final_test_auc:.4f}")
    return final_model, best_params

if __name__ == '__main__':
    # Example usage - would typically be called from a main orchestration script
    # For demonstration, creating dummy data
    num_samples = 1000
    num_meta_features = 3
    dummy_X_meta_train = np.random.rand(num_samples, num_meta_features)
    dummy_y_full = pd.Series(np.random.randint(0, 2, num_samples))
    dummy_X_test = pd.DataFrame(np.random.rand(num_samples // 4, num_meta_features))
    dummy_y_test = pd.Series(np.random.randint(0, 2, num_samples // 4))

    print("Starting dummy meta-learner tuning and training...")
    trained_mlp_model, tuned_params = tune_and_train_meta_learner(dummy_X_meta_train, dummy_y_full, dummy_X_test, dummy_y_test, n_trials=10)
    print("Dummy meta-learner tuning and training finished.")
