#!/usr/bin/env python3
"""
Particle Swarm Optimization (PSO) for Hyperparameter Tuning
Optimizes cardiovascular transformer hyperparameters
"""

import numpy as np
import random
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for hyperparameter tuning
    """
    
    def __init__(self, 
                 n_particles=30,
                 n_iterations=50,
                 w=0.7,  # Inertia weight
                 c1=1.5,  # Cognitive learning factor
                 c2=1.5,  # Social learning factor
                 random_state=42):
        
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.random_state = random_state
        
        # Set random seeds
        random.seed(random_state)
        np.random.seed(random_state)
        
        # History tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_params_history = []
        
    def optimize(self, objective_function, param_bounds, X, y, cv=5):
        """
        Run PSO optimization
        
        Args:
            objective_function: Function to optimize (should return fitness score)
            param_bounds: Dictionary of parameter bounds {param_name: (min, max)}
            X: Feature matrix
            y: Target vector
            cv: Cross-validation folds
        
        Returns:
            best_params: Best hyperparameters found
            best_fitness: Best fitness score
            history: Optimization history
        """
        print(f"🚀 Starting PSO Hyperparameter Optimization...")
        print(f"   Particles: {self.n_particles}")
        print(f"   Iterations: {self.n_iterations}")
        print(f"   Parameters to optimize: {list(param_bounds.keys())}")
        
        # Initialize particles
        particles = self._initialize_particles(param_bounds)
        velocities = self._initialize_velocities(param_bounds)
        
        # Initialize personal best positions and fitness
        personal_best_positions = particles.copy()
        personal_best_fitness = np.full(self.n_particles, -np.inf)
        
        # Initialize global best
        global_best_position = None
        global_best_fitness = -np.inf
        
        # Evaluate initial population
        print("   Evaluating initial population...")
        for i in range(self.n_particles):
            params = self._particle_to_params(particles[i], param_bounds)
            fitness = objective_function(params, X, y, cv)
            personal_best_fitness[i] = fitness
            
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particles[i].copy()
        
        print(f"   Initial best fitness: {global_best_fitness:.4f}")
        
        # Main optimization loop
        for iteration in range(self.n_iterations):
            # Update particles
            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = random.random(), random.random()
                
                cognitive_velocity = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                social_velocity = self.c2 * r2 * (global_best_position - particles[i])
                
                velocities[i] = (self.w * velocities[i] + 
                               cognitive_velocity + 
                               social_velocity)
                
                # Update position
                particles[i] += velocities[i]
                
                # Clamp to bounds
                particles[i] = np.clip(particles[i], 0, 1)
                
                # Evaluate new position
                params = self._particle_to_params(particles[i], param_bounds)
                fitness = objective_function(params, X, y, cv)
                
                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = particles[i].copy()
                    
                    # Update global best
                    if fitness > global_best_fitness:
                        global_best_fitness = fitness
                        global_best_position = particles[i].copy()
            
            # Record history
            self.best_fitness_history.append(global_best_fitness)
            self.avg_fitness_history.append(np.mean(personal_best_fitness))
            self.best_params_history.append(self._particle_to_params(global_best_position, param_bounds))
            
            # Print progress
            if iteration % 10 == 0:
                print(f"   Iteration {iteration:3d}: Best = {global_best_fitness:.4f}, "
                      f"Avg = {np.mean(personal_best_fitness):.4f}")
        
        # Get final best parameters
        best_params = self._particle_to_params(global_best_position, param_bounds)
        
        print(f"\n✅ PSO Optimization completed!")
        print(f"   Best fitness: {global_best_fitness:.4f}")
        print(f"   Best parameters: {best_params}")
        
        # Prepare history
        history = {
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'best_params_history': self.best_params_history
        }
        
        return best_params, global_best_fitness, history
    
    def _initialize_particles(self, param_bounds):
        """Initialize particle positions randomly"""
        n_params = len(param_bounds)
        particles = np.random.random((self.n_particles, n_params))
        return particles
    
    def _initialize_velocities(self, param_bounds):
        """Initialize particle velocities"""
        n_params = len(param_bounds)
        velocities = np.random.uniform(-0.1, 0.1, (self.n_particles, n_params))
        return velocities
    
    def _particle_to_params(self, particle, param_bounds):
        """Convert normalized particle position to actual parameter values"""
        params = {}
        param_names = list(param_bounds.keys())
        
        for i, param_name in enumerate(param_names):
            min_val, max_val = param_bounds[param_name]
            
            # Convert from [0,1] to [min_val, max_val]
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Integer parameters
                params[param_name] = int(min_val + particle[i] * (max_val - min_val))
            else:
                # Float parameters
                params[param_name] = min_val + particle[i] * (max_val - min_val)
        
        return params

def create_transformer_objective_function():
    """
    Create objective function for cardiovascular transformer hyperparameter optimization
    
    Returns:
        objective_function: Function that evaluates transformer hyperparameters
    """
    def objective_function(params, X, y, cv=5):
        """
        Objective function for transformer hyperparameter optimization
        
        Args:
            params: Dictionary of hyperparameters
            X: Feature matrix
            y: Target vector
            cv: Cross-validation folds
        
        Returns:
            fitness_score: Cross-validation accuracy
        """
        try:
            # Import here to avoid circular imports
            from cardiovascular_transformer import CardiovascularDiseasePredictor
            
            # Extract hyperparameters
            d_model = params.get('d_model', 128)
            nhead = params.get('nhead', 8)
            num_layers = params.get('num_layers', 4)
            dropout = params.get('dropout', 0.1)
            learning_rate = params.get('learning_rate', 1e-4)
            batch_size = params.get('batch_size', 32)
            epochs = params.get('epochs', 20)
            
            # Create transformer predictor
            predictor = CardiovascularDiseasePredictor(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs
            )
            
            # Prepare data
            train_loader, val_loader, _ = predictor.prepare_data(X, y, test_size=0.2)
            
            # Create and train model
            predictor.create_model(X.shape[1], num_classes=len(np.unique(y)))
            history = predictor.train_model(train_loader, val_loader)
            
            # Return validation accuracy as fitness
            fitness = history['best_val_acc']
            
            return fitness
            
        except Exception as e:
            # Return low fitness for failed configurations
            print(f"     Warning: Parameter evaluation failed: {e}")
            return 0.0
    
    return objective_function

def create_random_forest_objective_function():
    """
    Create objective function for Random Forest hyperparameter optimization
    
    Returns:
        objective_function: Function that evaluates RF hyperparameters
    """
    def objective_function(params, X, y, cv=5):
        """
        Objective function for Random Forest hyperparameter optimization
        
        Args:
            params: Dictionary of hyperparameters
            X: Feature matrix
            y: Target vector
            cv: Cross-validation folds
        
        Returns:
            fitness_score: Cross-validation accuracy
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # Extract hyperparameters
            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', None)
            min_samples_split = params.get('min_samples_split', 2)
            min_samples_leaf = params.get('min_samples_leaf', 1)
            max_features = params.get('max_features', 'sqrt')
            
            # Create Random Forest model
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=-1
            )
            
            # Perform cross-validation (use n_jobs=1 for Windows compatibility)
            scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
            fitness = np.mean(scores)
            
            return fitness
            
        except Exception as e:
            print(f"     Warning: Parameter evaluation failed: {e}")
            return 0.0
    
    return objective_function

def run_transformer_pso_optimization(X, y, n_particles=20, n_iterations=30):
    """
    Run PSO optimization for cardiovascular transformer hyperparameters
    
    Args:
        X: Feature matrix
        y: Target vector
        n_particles: Number of particles
        n_iterations: Number of iterations
    
    Returns:
        best_params: Best hyperparameters found
        best_fitness: Best fitness score
        history: Optimization history
    """
    print("="*70)
    print("PSO HYPERPARAMETER OPTIMIZATION FOR CARDIOVASCULAR TRANSFORMER")
    print("="*70)
    
    # Define parameter bounds for transformer
    transformer_bounds = {
        'd_model': (32, 256),      # Model dimension
        'nhead': (2, 16),          # Number of attention heads
        'num_layers': (1, 8),      # Number of transformer layers
        'dropout': (0.0, 0.5),     # Dropout rate
        'learning_rate': (1e-5, 1e-3),  # Learning rate
        'batch_size': (16, 128),   # Batch size
        'epochs': (10, 50)         # Training epochs
    }
    
    # Create PSO optimizer
    pso = ParticleSwarmOptimizer(
        n_particles=n_particles,
        n_iterations=n_iterations,
        w=0.7,
        c1=1.5,
        c2=1.5,
        random_state=42
    )
    
    # Create objective function
    objective_func = create_transformer_objective_function()
    
    # Run optimization
    best_params, best_fitness, history = pso.optimize(
        objective_func, transformer_bounds, X, y, cv=3
    )
    
    print(f"\n🎯 OPTIMIZATION RESULTS:")
    print(f"   Best fitness (validation accuracy): {best_fitness:.4f}")
    print(f"   Best hyperparameters:")
    for param, value in best_params.items():
        print(f"     {param}: {value}")
    
    return best_params, best_fitness, history

if __name__ == "__main__":
    print("Particle Swarm Optimization for Hyperparameter Tuning")
    print("This module provides PSO implementation for:")
    print("- Cardiovascular transformer hyperparameter optimization")
    print("- Random Forest hyperparameter optimization")
    print("- General hyperparameter search")
