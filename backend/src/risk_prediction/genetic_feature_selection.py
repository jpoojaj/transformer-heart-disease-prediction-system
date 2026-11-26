import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools, algorithms
import random
import warnings
warnings.filterwarnings('ignore')

class GeneticFeatureSelector:
    """
    Advanced Genetic Algorithm for Feature Selection using DEAP library
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 generations: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 tournament_size: int = 3,
                 random_state: int = 42):
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.random_state = random_state
        
        # Set random seeds
        random.seed(random_state)
        np.random.seed(random_state)
        
        # History tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.feature_count_history = []
        self.best_individual_history = []
        
        # DEAP setup
        self.setup_deap()
        
    def setup_deap(self):
        """Setup DEAP genetic algorithm components"""
        # Create fitness class (maximize)
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        # Create individual class
        if hasattr(creator, "Individual"):
            del creator.Individual
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Create toolbox
        self.toolbox = base.Toolbox()
        
        # Attribute generator
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        
        # Individual and population
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                            self.toolbox.attr_bool, n=0)  # n will be set later
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
    def evaluate_individual(self, individual):
        """
        Evaluate individual fitness
        Returns: single fitness value (accuracy - complexity_penalty)
        """
        # Convert binary list to feature mask
        feature_mask = np.array(individual, dtype=bool)
        
        # If no features selected, return very low fitness
        if not np.any(feature_mask):
            return (-1000.0,)
        
        # Get selected features
        X_selected = self.X[:, feature_mask]
        
        try:
            # Perform cross-validation (use n_jobs=1 for Windows compatibility)
            cv_scores = cross_val_score(
                self.model, X_selected, self.y, 
                cv=self.cv, scoring='accuracy', n_jobs=1
            )
            
            accuracy = np.mean(cv_scores)
            feature_count = np.sum(feature_mask)
            
            # Complexity penalty (encourage fewer features)
            complexity_penalty = feature_count / len(individual) * 0.1
            
            # Return single fitness value as tuple (DEAP requirement)
            fitness = accuracy - complexity_penalty
            
            return (fitness,)
            
        except Exception as e:
            return (0.0,)
    
    def fit(self, X, y, feature_names=None, model=None, cv=5):
        """
        Run genetic algorithm for feature selection
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            model: Base model (default: RandomForestClassifier)
            cv: Cross-validation folds
        """
        self.X = X
        self.y = y
        self.cv = cv
        
        # Set default model if none provided
        if model is None:
            self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            self.model = model
        
        # Set individual length to number of features
        n_features = X.shape[1]
        self.toolbox.individual.keywords['n'] = n_features
        
        print(f"Starting Genetic Algorithm for Feature Selection...")
        print(f"Dataset: {X.shape[0]} samples, {n_features} features")
        print(f"Population: {self.population_size}, Generations: {self.generations}")
        print(f"CV folds: {cv}")
        
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Track best individual
        best_individual = None
        best_fitness = -np.inf
        
        # Evolution loop
        for generation in range(self.generations):
            # Select and clone next generation individuals
            offspring = list(map(self.toolbox.clone, self.toolbox.select(pop, len(pop))))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < self.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            pop[:] = offspring
            
            # Get best individual of this generation
            gen_best = self.toolbox.select(pop, 1)[0]
            gen_best_fitness = gen_best.fitness.values[0]
            
            # Update best overall
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = gen_best[:]
            
            # Record history
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean([ind.fitness.values[0] for ind in pop]))
            
            # Count features in best individual
            feature_count = np.sum(best_individual)
            self.feature_count_history.append(feature_count)
            self.best_individual_history.append(best_individual[:])
            
            # Print progress
            if generation % 5 == 0:
                print(f"Generation {generation:3d}: Best = {best_fitness:.4f}, "
                      f"Avg = {np.mean([ind.fitness.values[0] for ind in pop]):.4f}, "
                      f"Features = {feature_count}")
        
        # Store results
        self.best_individual = best_individual
        self.best_fitness = best_fitness
        self.selected_features_mask = np.array(best_individual, dtype=bool)
        self.selected_features = np.arange(n_features)[self.selected_features_mask]
        
        print(f"\nOptimization completed!")
        print(f"Best fitness: {best_fitness:.4f}")
        print(f"Selected features: {np.sum(self.selected_features_mask)} out of {n_features}")
        
        # Return selected features, best fitness, and history
        selected_features = [feature_names[i] for i in self.selected_features]
        history = {
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'feature_count_history': self.feature_count_history
        }
        
        return selected_features, best_fitness, history
    
    def transform(self, X):
        """Transform data using selected features"""
        return X[:, self.selected_features_mask]
    
    def get_support(self):
        """Get boolean mask of selected features"""
        return self.selected_features_mask
    
    def get_selected_features(self):
        """Get indices of selected features"""
        return self.selected_features
    
    def get_feature_names(self, feature_names):
        """Get names of selected features"""
        return feature_names[self.selected_features_mask]
    
    def plot_evolution(self, save_path='genetic_feature_selection_evolution.png'):
        """Plot evolution progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Fitness evolution
        axes[0, 0].plot(self.best_fitness_history, 'b-', label='Best Fitness', linewidth=2)
        axes[0, 0].plot(self.avg_fitness_history, 'r--', label='Average Fitness', linewidth=1)
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness Score')
        axes[0, 0].set_title('Fitness Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Feature count evolution
        axes[0, 1].plot(self.feature_count_history, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Number of Selected Features')
        axes[0, 1].set_title('Feature Count Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Best fitness over time
        axes[1, 0].plot(self.best_fitness_history, 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Best Fitness Score')
        axes[1, 0].set_title('Best Fitness Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature selection comparison
        total_features = len(self.best_individual)
        selected_features = np.sum(self.selected_features_mask)
        feature_counts = [total_features, selected_features]
        labels = ['Total Features', 'Selected Features']
        colors = ['#ff7f0e', '#2ca02c']
        
        axes[1, 1].pie(feature_counts, labels=labels, colors=colors, 
                       autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Feature Reduction')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Evolution plot saved as '{save_path}'")
    
    def plot_feature_importance(self, X, y, feature_names=None, save_path='feature_importance.png'):
        """Plot feature importance of selected features"""
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Train final model with selected features
        X_selected = self.transform(X)
        final_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        final_model.fit(X_selected, y)
        
        # Get feature importance
        selected_feature_names = self.get_feature_names(feature_names)
        feature_importance = pd.DataFrame({
            'feature': selected_feature_names,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Feature importance bar plot
        plt.subplot(2, 1, 1)
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Feature Importance Ranking (Selected Features)')
        plt.xlabel('Importance')
        
        # Feature importance over time
        plt.subplot(2, 1, 2)
        importance_matrix = np.array(self.best_individual_history)
        plt.imshow(importance_matrix.T, aspect='auto', cmap='Blues')
        plt.xlabel('Generation')
        plt.ylabel('Feature Index')
        plt.title('Feature Selection Evolution')
        plt.colorbar(label='Selected (1) / Not Selected (0)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Feature importance plot saved as '{save_path}'")
        
        return feature_importance

def run_genetic_feature_selection(X, y, 
                                 population_size=100, 
                                 generations=50,
                                 model=None,
                                 cv=5):
    """
    Run genetic algorithm for feature selection
    
    Args:
        X: Feature matrix
        y: Target vector
        population_size: Population size for GA
        generations: Number of generations
        model: Base model (default: RandomForestClassifier)
        cv: Cross-validation folds
    
    Returns:
        GeneticFeatureSelector: Trained feature selector
    """
    
    # Initialize and run genetic algorithm
    ga_selector = GeneticFeatureSelector(
        population_size=population_size,
        generations=generations,
        mutation_rate=0.1,
        crossover_rate=0.8,
        tournament_size=3,
        random_state=42
    )
    
    # Fit the selector
    ga_selector.fit(X, y, model=model, cv=cv)
    
    # Plot evolution
    ga_selector.plot_evolution()
    
    return ga_selector

if __name__ == "__main__":
    print("Genetic Feature Selection Module using DEAP")
    print("This module provides advanced genetic algorithm for feature selection")
    print("Features:")
    print("- Multi-objective fitness (accuracy, feature count, complexity)")
    print("- Advanced genetic operators (tournament selection, two-point crossover)")
    print("- Comprehensive visualization and analysis")
    print("- Integration with scikit-learn pipeline")
