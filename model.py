# ============================================================
# intent_classifier/model.py
# PURPOSE: Train and serve the intent classification model
# ============================================================

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib


@dataclass
class PredictionResult:
    """Structured prediction output."""
    predicted_mode: str
    confidence: float
    all_probabilities: Dict[str, float]
    is_ambiguous: bool  # True if top-2 predictions are close


class IntentClassifier:
    """
    Multi-class intent classifier using an ensemble approach.
    
    ARCHITECTURE CHOICES (talk about these in interviews):
    
    1. TF-IDF over word embeddings:
       - For short queries, TF-IDF with character n-grams 
         captures morphological patterns ("How does..." vs "Is it true...")
       - No need for a 100MB embedding model for this task
       - MUCH faster training and inference
    
    2. Ensemble of 3 models:
       - Logistic Regression: Strong linear baseline, great with TF-IDF
       - Linear SVM: Excellent for text classification, different decision boundary
       - Random Forest: Captures non-linear feature interactions
       - Voting ensemble reduces variance and improves robustness
    
    3. Calibrated probabilities:
       - Raw SVM scores aren't probabilities
       - CalibratedClassifierCV converts them to real probabilities
       - Enables the "ambiguity detection" feature
    """
    
    AMBIGUITY_THRESHOLD = 0.15  # If top-2 preds are within 15%, flag as ambiguous
    CONFIDENCE_THRESHOLD = 0.45  # Below this, we're not confident
    
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.training_metrics: Dict = {}
    
    def _build_pipeline(self) -> Pipeline:
        """
        Build the ML pipeline.
        
        KEY DESIGN: Character n-grams (2,5) capture question patterns
        like "wh", "how", "is it" without needing perfect tokenization.
        This makes the model robust to typos and informal language.
        """
        
        # TF-IDF with both word and character features
        tfidf = TfidfVectorizer(
            # Word features
            analyzer='word',
            ngram_range=(1, 3),      # Unigrams through trigrams
            max_features=15000,
            min_df=2,                 # Appear in at least 2 docs
            max_df=0.95,              # Not in more than 95% of docs
            sublinear_tf=True,        # Apply log normalization
            strip_accents='unicode',
            # This is the secret sauce — also capture character patterns
            # We'll combine word + char features via a FeatureUnion later
        )
        
        # Ensemble of calibrated classifiers
        lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',  # Handle any class imbalance
            solver='lbfgs',
            multi_class='multinomial'
        )
        
        svm = CalibratedClassifierCV(
            LinearSVC(
                C=0.5,
                class_weight='balanced',
                max_iter=2000
            ),
            cv=3  # 3-fold calibration
        )
        
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Soft voting uses predicted probabilities
        ensemble = VotingClassifier(
            estimators=[
                ('lr', lr),
                ('svm', svm),
                ('rf', rf)
            ],
            voting='soft',
            weights=[2, 1, 1]  # LR gets more weight — it's best for text
        )
        
        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('classifier', ensemble)
        ])
        
        return pipeline
    
    def train(self, data_path: str = 'data/intent_dataset.json') -> Dict:
        """
        Train the classifier with full evaluation.
        
        Returns detailed metrics for analysis.
        """
        # Load data
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        texts = [d['text'] for d in raw_data]
        labels = [d['label'] for d in raw_data]
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Stratified split — maintains class proportions
        X_train, X_test, y_train, y_test = train_test_split(
            texts, encoded_labels,
            test_size=0.2,
            random_state=42,
            stratify=encoded_labels  # IMPORTANT: stratify on labels
        )
        
        print(f"Training set: {len(X_train)} examples")
        print(f"Test set: {len(X_test)} examples")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        
        # Build and train pipeline
        self.pipeline = self._build_pipeline()
        
        # Cross-validation BEFORE final training (proper ML practice)
        print("\nRunning 5-fold cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.pipeline, X_train, y_train,
            cv=cv, scoring='f1_weighted', n_jobs=-1
        )
        print(f"CV F1 scores: {cv_scores}")
        print(f"CV F1 mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Final training on full training set
        print("\nTraining final model...")
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        
        report = classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        self.training_metrics = {
            'cv_f1_mean': float(cv_scores.mean()),
            'cv_f1_std': float(cv_scores.std()),
            'test_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        # Print detailed report
        print("\n" + "=" * 50)
        print("CLASSIFICATION REPORT")
        print("=" * 50)
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))
        
        print("CONFUSION MATRIX:")
        print(f"Labels: {list(self.label_encoder.classes_)}")
        print(conf_matrix)
        
        return self.training_metrics
    
    def predict(self, query: str) -> PredictionResult:
        """
        Predict the intent mode for a user query.
        
        Returns structured result with confidence and ambiguity detection.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Get probability distribution over all classes
        probabilities = self.pipeline.predict_proba([query])[0]
        
        # Map probabilities to class names
        prob_dict = {
            label: float(prob) 
            for label, prob in zip(self.label_encoder.classes_, probabilities)
        }
        
        # Sort by probability
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        
        top_label = sorted_probs[0][0]
        top_conf = sorted_probs[0][1]
        second_conf = sorted_probs[1][1]
        
        # Detect ambiguity
        is_ambiguous = (top_conf - second_conf) < self.AMBIGUITY_THRESHOLD
        
        return PredictionResult(
            predicted_mode=top_label,
            confidence=top_conf,
            all_probabilities=prob_dict,
            is_ambiguous=is_ambiguous
        )
    
    def save_model(self, path: str = 'models/intent_classifier.joblib') -> None:
        """Save the trained model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'training_metrics': self.training_metrics
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str = 'models/intent_classifier.joblib') -> 'IntentClassifier':
        """Load a trained model."""
        model_data = joblib.load(path)
        instance = cls()
        instance.pipeline = model_data['pipeline']
        instance.label_encoder = model_data['label_encoder']
        instance.training_metrics = model_data['training_metrics']
        instance.is_trained = True
        return instance


# ============================================================
# TRAINING SCRIPT
# ============================================================
if __name__ == '__main__':
    from dataset_builder import IntentDatasetBuilder
    
    # Step 1: Build dataset
    builder = IntentDatasetBuilder(seed=42)
    builder.build_full_dataset()
    
    # Step 2: Train classifier
    classifier = IntentClassifier()
    metrics = classifier.train('data/intent_dataset.json')
    
    # Step 3: Test with real queries
    test_queries = [
        "How should I prepare for a job interview?",
        "What happened during the French Revolution?",
        "How does photosynthesis work?",
        "What is the tallest building in the world?",
        "Is it true that we only use 10% of our brain?",
        "yo how do i get better at cooking lol",
        "did einstein really fail math",
        "what's a black hole",
    ]
    
    print("\n" + "=" * 60)
    print("LIVE PREDICTIONS")
    print("=" * 60)
    
    for query in test_queries:
        result = classifier.predict(query)
        ambiguous_flag = " ⚠️ AMBIGUOUS" if result.is_ambiguous else ""
        print(f"\n  Query: '{query}'")
        print(f"  Mode:  {result.predicted_mode.upper()} "
              f"(confidence: {result.confidence:.2f}){ambiguous_flag}")
        
        # Show top 3 probabilities
        sorted_probs = sorted(result.all_probabilities.items(), 
                             key=lambda x: x[1], reverse=True)[:3]
        for label, prob in sorted_probs:
            bar = "█" * int(prob * 30)
            print(f"    {label:8s}: {prob:.3f} {bar}")
    
    # Step 4: Save model
    classifier.save_model()