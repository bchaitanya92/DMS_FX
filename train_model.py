import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import joblib
import os
import json
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_features(df):
    """Prepare features for training"""
    logger.info("Preparing features...")
    # Extract time-based features
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    
    # Create location encoding
    location_dummies = pd.get_dummies(df['location'], prefix='location')
    
    # Select features for training
    feature_cols = ['temperature', 'wind_speed', 'rainfall', 'humidity', 'pressure',
                   'hour', 'month', 'day_of_week']
    
    # Combine all features
    X = pd.concat([df[feature_cols], location_dummies], axis=1)
    y = df['disaster_type']
    
    logger.info(f"Created {len(feature_cols)} base features and {len(location_dummies.columns)} location features")
    return X, y

def train_model(X_train, X_test, y_train, y_test):
    """Train the ensemble model"""
    try:
        logger.info("Initializing models...")
        # Initialize base models with more conservative parameters
        rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
        xgb_model = xgb.XGBClassifier(max_depth=4, n_estimators=50, random_state=42)
        lgb_model = lgb.LGBMClassifier(max_depth=4, n_estimators=50, random_state=42, verbose=-1)
        gb = GradientBoostingClassifier(max_depth=4, n_estimators=50, random_state=42)
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('gb', gb)
            ],
            voting='soft'
        )
        
        # Train the model
        logger.info("Training ensemble model...")
        ensemble.fit(X_train, y_train)
        
        # Make predictions
        logger.info("Making predictions on test set...")
        y_pred = ensemble.predict(X_test)
        
        # Calculate metrics
        logger.info("Calculating performance metrics...")
        report = classification_report(y_test, y_pred)
        logger.info("\nClassification Report:\n" + report)
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=sorted(y_test.unique()),
            columns=sorted(y_test.unique())
        )
        logger.info("\nConfusion Matrix:\n" + str(cm_df))
        
        return ensemble, cm_df
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

def save_model_and_metrics(model, scaler, metrics, feature_names):
    """Save the trained model and its metrics"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        logger.info("Saving model and scaler...")
        # Save the model and scaler
        joblib.dump(model, 'models/ensemble_model.joblib')
        joblib.dump(scaler, 'models/scaler.joblib')
        
        logger.info("Saving feature names...")
        # Save feature names
        with open('models/feature_names.json', 'w') as f:
            json.dump(list(feature_names), f)
        
        logger.info("Saving metrics...")
        # Save metrics
        metrics['timestamp'] = datetime.now().isoformat()
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info("Successfully saved all model artifacts")
    except Exception as e:
        logger.error(f"Error saving model artifacts: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Load the historical data
        logger.info("Loading historical weather data...")
        df = pd.read_csv('data/historical_weather_data.csv')
        logger.info(f"Loaded {len(df)} records")
        
        # Prepare features
        X, y = prepare_features(df)
        
        # Split the data
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model, confusion_matrix = train_model(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Calculate additional metrics
        logger.info("Calculating final metrics...")
        y_pred = model.predict(X_test_scaled)
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': confusion_matrix.to_dict(),
            'n_samples': len(df),
            'training_period': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'class_distribution': y.value_counts().to_dict()
        }
        
        # Save everything
        save_model_and_metrics(model, scaler, metrics, X.columns)
        
        logger.info("\nTraining completed successfully!")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise 