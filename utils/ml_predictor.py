import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import logging
from .weather_fetcher import WeatherDataFetcher
from .data_collector import DisasterDataCollector
import joblib
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DisasterPredictor:
    def __init__(self):
        """Initialize the DisasterPredictor with advanced models"""
        logger.info("Initializing DisasterPredictor...")
        
        # Initialize models for each disaster type
        self.models = {
            'flood': None,
            'cyclone': None,
            'heatwave': None,
            'storm': None
        }
        
        # Initialize components
        self.scaler = StandardScaler()
        self.weather_fetcher = WeatherDataFetcher()
        self.data_collector = DisasterDataCollector()
        
        # Define feature sets
        self.base_features = [
            'temperature', 'wind_speed', 'rainfall', 'humidity', 
            'pressure', 'soil_moisture'
        ]
        self.geo_features = [
            'elevation', 'population_density', 'distance_to_water'
        ]
        self.derived_features = [
            'temp_humidity_index', 'rain_wind_index'
        ]
        
        self.all_features = self.base_features + self.geo_features + self.derived_features
        
        # Load or train models
        self._load_or_train_models()
        logger.info("DisasterPredictor initialized successfully")

    def _load_or_train_models(self):
        """Load existing models or train new ones"""
        try:
            if self._models_exist():
                self._load_models()
            else:
                self._train_new_models()
        except Exception as e:
            logger.error(f"Error in model initialization: {str(e)}")
            self._train_new_models()

    def _models_exist(self):
        """Check if trained models exist"""
        required_files = [
            'models/flood_model.joblib',
            'models/cyclone_model.joblib',
            'models/heatwave_model.joblib',
            'models/storm_model.joblib',
            'models/scaler.joblib'
        ]
        return all(os.path.exists(f) for f in required_files)

    def _load_models(self):
        """Load trained models from files"""
        try:
            for disaster_type in self.models.keys():
                model_path = f'models/{disaster_type}_model.joblib'
                self.models[disaster_type] = joblib.load(model_path)
            self.scaler = joblib.load('models/scaler.joblib')
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def _train_new_models(self):
        """Train new models using collected data"""
        try:
            # Collect and prepare training data
            df = self.data_collector.collect_training_data()
            
            if df is None or df.empty:
                raise ValueError("No training data available")

            # Prepare features
            X = df[self.all_features]
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models for each disaster type
            for disaster_type in self.models.keys():
                logger.info(f"Training model for {disaster_type}")
                
                # Prepare target variable
                y = df[f'is_{disaster_type}']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Initialize model with optimized parameters
                model = GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                cv_scores = cross_val_score(model, X_scaled, y, cv=5)
                
                # Calculate ROC AUC score
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Log performance metrics
                logger.info(f"{disaster_type.capitalize()} Model Performance:")
                logger.info(f"Train Score: {train_score:.3f}")
                logger.info(f"Test Score: {test_score:.3f}")
                logger.info(f"CV Scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                logger.info(f"ROC AUC Score: {roc_auc:.3f}")
                
                # Store model
                self.models[disaster_type] = model
            
            # Save models and scaler
            os.makedirs('models', exist_ok=True)
            for disaster_type, model in self.models.items():
                joblib.dump(model, f'models/{disaster_type}_model.joblib')
            joblib.dump(self.scaler, 'models/scaler.joblib')
            
            logger.info("New models trained and saved successfully")
            
        except Exception as e:
            logger.error(f"Error training new models: {str(e)}")
            raise

    def predict_with_weather(self, location):
        """Predict potential disasters based on current weather data"""
        try:
            # Get current conditions including weather and geographical data
            conditions = self.data_collector.get_current_conditions(location)
            
            if not conditions:
                raise ValueError(f"Could not get conditions for {location}")
            
            # Prepare features
            features = np.array([[
                conditions.get(feature, 0) for feature in self.all_features
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make predictions for each disaster type
            predictions = {}
            for disaster_type, model in self.models.items():
                # Get probability of disaster
                prob = model.predict_proba(features_scaled)[0][1]
                predictions[disaster_type] = prob
            
            # Get disaster types with high probability
            high_risk_disasters = [
                dtype for dtype, prob in predictions.items()
                if prob >= 0.5
            ]
            
            # Calculate overall risk
            max_prob = max(predictions.values()) if predictions else 0
            
            # Get feature importance for explainability
            feature_importance = self._get_feature_importance(location, predictions)
            
            return {
                'probability': float(max_prob),
                'disaster_types': high_risk_disasters,
                'predictions': predictions,
                'weather_data': conditions,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {
                'probability': 0.0,
                'disaster_types': [],
                'predictions': {},
                'weather_data': None,
                'feature_importance': {}
            }

    def _get_feature_importance(self, location, predictions):
        """Calculate feature importance for the current prediction"""
        try:
            feature_importance = {}
            
            # Get the most likely disaster type
            max_disaster = max(predictions.items(), key=lambda x: x[1])[0]
            model = self.models[max_disaster]
            
            # Get feature importance from the model
            importance = model.feature_importances_
            
            # Create feature importance dictionary
            for feature, imp in zip(self.all_features, importance):
                feature_importance[feature] = float(imp)
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}

    def get_model_info(self):
        """Get information about the current models"""
        return {
            'last_updated': datetime.now().isoformat(),
            'model_types': {
                disaster_type: {
                    'type': type(model).__name__,
                    'features': self.all_features,
                    'parameters': model.get_params()
                }
                for disaster_type, model in self.models.items()
            },
            'feature_sets': {
                'base_features': self.base_features,
                'geo_features': self.geo_features,
                'derived_features': self.derived_features
            }
        } 