import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import logging
from .weather_fetcher import WeatherDataFetcher
from .time_series_handler import TimeSeriesHandler
import joblib
import os
from sklearn.metrics import roc_auc_score
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DisasterPredictor:
    def __init__(self):
        """Initialize the disaster predictor with multiple models for different disaster types"""
        self.models = {}
        self.scaler = StandardScaler()
        self.base_features = [
            'temperature', 'wind_speed', 'rainfall', 'humidity', 
            'pressure', 'soil_moisture'
        ]
        self.geo_features = [
            'elevation', 'population_density', 'distance_to_water',
            'land_use_index'
        ]
        self.derived_features = [
            'temp_humidity_index', 'rain_wind_index'
        ]
        self.all_features = self.base_features + self.geo_features + self.derived_features
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Initialize models for different disaster types
        self._load_or_train_models()
        
        # Track last update time and current accuracy
        self.last_update = datetime.now()
        self.current_accuracy = 0.0

    def _load_or_train_models(self):
        """Load existing models or train new ones if they don't exist"""
        disaster_types = ['flood', 'cyclone', 'heatwave', 'storm']
        
        try:
            if self._models_exist():
                logger.info("Loading existing models...")
                for disaster_type in disaster_types:
                    model_path = f'models/{disaster_type}_model.joblib'
                    self.models[disaster_type] = joblib.load(model_path)
                self.scaler = joblib.load('models/scaler.joblib')
                logger.info("Models loaded successfully")
            else:
                logger.info("Training new models...")
                self._train_new_models()
        except Exception as e:
            logger.error(f"Error in model initialization: {str(e)}")
            self._train_new_models()

    def _models_exist(self):
        """Check if trained models exist"""
        model_files = [
            'models/flood_model.joblib',
            'models/cyclone_model.joblib',
            'models/heatwave_model.joblib',
            'models/storm_model.joblib',
            'models/scaler.joblib'
        ]
        return all(os.path.exists(f) for f in model_files)

    def _train_new_models(self):
        """Train new models for each disaster type"""
        try:
            # Initialize models with optimized parameters
            base_params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42
            }
            
            model_params = {
                'flood': {'n_estimators': 250, 'max_depth': 6},
                'cyclone': {'n_estimators': 300, 'max_depth': 7},
                'heatwave': {'n_estimators': 200, 'max_depth': 5},
                'storm': {'n_estimators': 250, 'max_depth': 6}
            }
            
            for disaster_type, params in model_params.items():
                # Create model with combined parameters
                model_params = {**base_params, **params}
                self.models[disaster_type] = GradientBoostingClassifier(**model_params)
                
                # Load training data
                data = self._load_training_data(disaster_type)
                if data is not None:
                    X = data[self.all_features]
                    y = data[f'{disaster_type}_probability'] > 0.5
                    
                    # Scale features
                    X_scaled = self.scaler.fit_transform(X)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=0.2, random_state=42
                    )
                    
                    # Train model
                    self.models[disaster_type].fit(X_train, y_train)
                    
                    # Evaluate model
                    train_score = self.models[disaster_type].score(X_train, y_train)
                    test_score = self.models[disaster_type].score(X_test, y_test)
                    cv_scores = cross_val_score(
                        self.models[disaster_type], X_scaled, y, cv=5
                    )
                    roc_auc = roc_auc_score(y_test, self.models[disaster_type].predict_proba(X_test)[:, 1])
                    
                    logger.info(f"{disaster_type.title()} Model Performance:")
                    logger.info(f"Train Score: {train_score:.3f}")
                    logger.info(f"Test Score: {test_score:.3f}")
                    logger.info(f"CV Scores Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                    logger.info(f"ROC AUC Score: {roc_auc:.3f}")
                    
                    # Save model
                    joblib.dump(self.models[disaster_type], f'models/{disaster_type}_model.joblib')
            
            # Save scaler
            joblib.dump(self.scaler, 'models/scaler.joblib')
            logger.info("Models trained and saved successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise

    def _load_training_data(self, disaster_type):
        """Load and preprocess training data for a specific disaster type"""
        try:
            # Load data from CSV files
            data_files = [f for f in os.listdir('data') if f.startswith('training_data_')]
            
            if not data_files:
                logger.error("No training data files found")
                return None
            
            # Combine data from all locations
            dfs = []
            for file in data_files:
                df = pd.read_csv(f'data/{file}')
                dfs.append(df)
            
            combined_data = pd.concat(dfs, ignore_index=True)
            
            # Ensure all required features are present
            for feature in self.all_features:
                if feature not in combined_data.columns:
                    combined_data[feature] = 0
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            return None

    def predict_with_weather(self, location):
        """Make predictions based on current weather conditions"""
        try:
            # Get current conditions
            from utils.data_collector import DisasterDataCollector
            collector = DisasterDataCollector(self.weather_fetcher)
            conditions = collector.get_current_conditions(location)
            
            if conditions is None:
                logger.error("Failed to get current conditions")
                return None
            
            # Prepare feature vector
            features = np.array([[
                conditions.get(feature, 0) for feature in self.all_features
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make predictions for each disaster type
            predictions = {}
            for disaster_type, model in self.models.items():
                prob = model.predict_proba(features_scaled)[0][1]
                predictions[disaster_type] = prob
            
            # Get feature importance
            feature_importance = self._get_feature_importance()
            
            return {
                'predictions': predictions,
                'feature_importance': feature_importance,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None

    def _get_feature_importance(self):
        """Get feature importance for current predictions"""
        try:
            importance_dict = {}
            for disaster_type, model in self.models.items():
                importance = model.feature_importances_
                importance_dict[disaster_type] = dict(zip(self.all_features, importance))
            return importance_dict
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}

    def get_risk_factors(self, weather_data):
        """Calculate risk factors based on weather conditions"""
        risk_factors = {
            'temperature': {
                'risk': 'high' if weather_data['temperature'] > 35 else 'medium' if weather_data['temperature'] > 30 else 'low',
                'value': weather_data['temperature']
            },
            'wind_speed': {
                'risk': 'high' if weather_data['wind_speed'] > 20 else 'medium' if weather_data['wind_speed'] > 15 else 'low',
                'value': weather_data['wind_speed']
            },
            'rainfall': {
                'risk': 'high' if weather_data['rainfall'] > 50 else 'medium' if weather_data['rainfall'] > 25 else 'low',
                'value': weather_data['rainfall']
            }
        }
        return risk_factors

    def get_update_status(self):
        """Get the update status of the model"""
        # This method should be implemented to return the current status of the model
        # For now, we'll return a placeholder
        return {
            'last_update_time': self.last_update.isoformat(),
            'current_accuracy': self.current_accuracy
        }

status = DisasterPredictor().get_update_status()
print(f"Last update: {status['last_update_time']}")
print(f"Current accuracy: {status['current_accuracy']}") 