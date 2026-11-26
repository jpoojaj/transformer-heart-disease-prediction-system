#!/usr/bin/env python3
"""
Cardiovascular Disease Transformer
Specialized transformer for predicting cardiovascular disease risk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class CardiovascularTransformer(nn.Module):
    """
    Transformer model specifically designed for cardiovascular disease prediction
    """
    
    def __init__(self, 
                 num_features,
                 num_classes=2,
                 d_model=128,
                 nhead=8,
                 num_layers=4,
                 dropout=0.1,
                 max_seq_length=50):
        
        super(CardiovascularTransformer, self).__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Feature embedding layer for cardiovascular features
        self.feature_embedding = nn.Linear(num_features, d_model)
        
        # Positional encoding for tabular cardiovascular data
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_length, d_model))
        
        # Transformer encoder for learning feature interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers for cardiovascular disease classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass for cardiovascular data
        
        Args:
            x: Input tensor of shape (batch_size, num_features)
        
        Returns:
            logits: Output logits for cardiovascular disease prediction
        """
        batch_size = x.size(0)
        
        # Embed cardiovascular features to d_model dimensions
        # Shape: (batch_size, num_features) -> (batch_size, num_features, d_model)
        x_embedded = self.feature_embedding(x).unsqueeze(1)
        
        # Add positional encoding (ensure same dtype)
        pos_encoding = self.pos_encoding[:, :x_embedded.size(1), :].to(x_embedded.dtype)
        x_embedded = x_embedded + pos_encoding
        
        # Pass through transformer to learn feature interactions
        # Shape: (batch_size, num_features, d_model) -> (batch_size, num_features, d_model)
        transformer_output = self.transformer(x_embedded)
        
        # Global pooling across features
        # Shape: (batch_size, num_features, d_model) -> (batch_size, d_model)
        pooled = self.global_pool(transformer_output.transpose(1, 2)).squeeze(-1)
        
        # Classification for cardiovascular disease
        logits = self.classifier(pooled)
        
        return logits

class CardiovascularDiseasePredictor:
    """
    Complete transformer pipeline for cardiovascular disease prediction
    """
    
    def __init__(self, 
                 d_model=128,
                 nhead=8,
                 num_layers=4,
                 dropout=0.1,
                 learning_rate=1e-4,
                 batch_size=32,
                 epochs=50,
                 device='auto'):
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        Prepare cardiovascular data for transformer training
        
        Args:
            X: Feature matrix (age, gender, height, weight, blood pressure, etc.)
            y: Target vector (cardiovascular disease: 0=no, 1=yes)
            test_size: Fraction of data for testing
            random_state: Random seed
        
        Returns:
            train_loader, val_loader, test_loader: Data loaders
        """
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=test_size, random_state=random_state, stratify=y_temp
        )
        
        # Scale features (important for transformer performance)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to numpy arrays if they are pandas Series/DataFrame
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        if hasattr(y_val, 'values'):
            y_val = y_val.values
        if hasattr(y_test, 'values'):
            y_test = y_test.values
            
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def create_model(self, num_features, num_classes=2):
        """Create the cardiovascular transformer model"""
        self.model = CardiovascularTransformer(
            num_features=num_features,
            num_classes=num_classes,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        print(f"Cardiovascular Transformer created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        return self.model
    
    def train_model(self, train_loader, val_loader, num_classes=2):
        """Train the cardiovascular transformer model"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10
        
        print("Starting cardiovascular disease prediction training...")
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = correct / total
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_cardiovascular_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d}/{self.epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_cardiovascular_model.pth'))
        
        print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }
    
    def evaluate_model(self, test_loader):
        """Evaluate the trained cardiovascular model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        auc = roc_auc_score(all_targets, [prob[1] for prob in all_probabilities])
        
        print("Cardiovascular Disease Prediction Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions, 
                                  target_names=['No Disease', 'Cardiovascular Disease']))
        
        return accuracy, auc, all_predictions, all_targets, all_probabilities
    
    def predict(self, X):
        """Make cardiovascular disease predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def get_feature_importance(self, X, y):
        """Get feature importance for cardiovascular disease prediction"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # This is a simplified feature importance method
        # In practice, you might want to use more sophisticated methods
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            # Get attention weights (simplified)
            # You would need to modify the model to extract attention weights
            pass
        
        # For now, return uniform importance
        num_features = X.shape[1]
        return np.ones(num_features) / num_features

def run_cardiovascular_experiment(X, y, 
                                 d_model=128, 
                                 nhead=8, 
                                 num_layers=4,
                                 epochs=50):
    """
    Run complete cardiovascular disease prediction experiment
    
    Args:
        X: Feature matrix (cardiovascular features)
        y: Target vector (cardiovascular disease: 0=no, 1=yes)
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        epochs: Number of training epochs
    
    Returns:
        CardiovascularDiseasePredictor: Trained cardiovascular predictor
    """
    print("="*60)
    print("CARDIOVASCULAR DISEASE PREDICTION EXPERIMENT")
    print("="*60)
    
    # Initialize cardiovascular predictor
    predictor = CardiovascularDiseasePredictor(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        epochs=epochs
    )
    
    # Prepare data
    print("Preparing cardiovascular data...")
    train_loader, val_loader, test_loader = predictor.prepare_data(X, y)
    
    # Create model
    print("Creating cardiovascular transformer model...")
    predictor.create_model(X.shape[1], num_classes=len(np.unique(y)))
    
    # Train model
    print("Training cardiovascular disease prediction model...")
    history = predictor.train_model(train_loader, val_loader)
    
    # Evaluate model
    print("Evaluating cardiovascular disease prediction model...")
    test_acc, test_auc, predictions, targets, probabilities = predictor.evaluate_model(test_loader)
    
    print(f"\nCardiovascular Disease Prediction Experiment completed!")
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Final test AUC-ROC: {test_auc:.4f}")
    
    return predictor, history

if __name__ == "__main__":
    print("Cardiovascular Disease Transformer")
    print("This module provides transformer implementation for cardiovascular disease prediction")
    print("Features:")
    print("- Custom transformer architecture for cardiovascular data")
    print("- Feature embedding and positional encoding")
    print("- Multi-head self-attention for feature interaction learning")
    print("- Comprehensive cardiovascular disease prediction pipeline")
    print("- Integration with GA feature selection")

