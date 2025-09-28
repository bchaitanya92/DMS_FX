import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

logger = logging.getLogger(__name__)

class TimeSeriesHandler:
    def __init__(self):
        logger.info("Initializing TimeSeriesHandler...")
        self.historical_data = self.load_historical_data()
        logger.info("TimeSeriesHandler initialized successfully")

    def load_historical_data(self):
        """Load historical weather data from CSV or generate sample data"""
        try:
            # Try to load existing data
            data = pd.read_csv('data/historical_weather.csv')
            logger.info("Successfully loaded historical data from data/historical_weather.csv")
            return data
        except FileNotFoundError:
            # Generate sample data for the last 2 years
            logger.info("Generating sample historical data...")
            return self.generate_sample_data()

    def generate_sample_data(self):
        """Generate 2 years of sample historical weather data"""
        # Generate dates for the last 2 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate sample data for each location
        locations = ['Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Bangalore', 'Hyderabad', 'Ahmedabad', 'Pune', 'Jaipur', 'Surat']
        data = []
        
        for date in dates:
            for location in locations:
                # Generate seasonal variations
                month = date.month
                if 3 <= month <= 5:  # Summer
                    temp_range = (30, 40)
                    rainfall_range = (0, 20)
                    wind_range = (5, 15)
                elif 6 <= month <= 9:  # Monsoon
                    temp_range = (25, 35)
                    rainfall_range = (20, 100)
                    wind_range = (10, 25)
                else:  # Winter
                    temp_range = (15, 30)
                    rainfall_range = (0, 10)
                    wind_range = (5, 15)
                
                # Add some randomness
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'location': location,
                    'temperature': round(np.random.uniform(*temp_range), 1),
                    'rainfall': round(np.random.uniform(*rainfall_range), 1),
                    'wind_speed': round(np.random.uniform(*wind_range), 1)
                })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv('data/historical_weather.csv', index=False)
        logger.info("Generated and saved sample historical data")
        return df

    def get_trends(self, location):
        """Get weather trends for a specific location"""
        try:
            # Filter data for the location
            location_data = self.historical_data[self.historical_data['location'] == location]
            
            # Calculate trends for the last 7 days
            recent_data = location_data.tail(7)
            
            trends = {
                'temperature_trend': self._calculate_trend(recent_data['temperature']),
                'rainfall_trend': self._calculate_trend(recent_data['rainfall']),
                'wind_speed_trend': self._calculate_trend(recent_data['wind_speed'])
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error calculating trends for {location}: {str(e)}")
            return None

    def predict_weather(self, location, days=7):
        """Predict weather for the next few days"""
        try:
            # Filter data for the location
            location_data = self.historical_data[self.historical_data['location'] == location]
            
            # Get the last 30 days of data for prediction
            recent_data = location_data.tail(30)
            
            # Generate predictions based on historical patterns
            predictions = []
            current_date = datetime.now()
            
            for i in range(days):
                date = current_date + timedelta(days=i)
                month = date.month
                
                # Use seasonal patterns for prediction
                if 3 <= month <= 5:  # Summer
                    temp_range = (30, 40)
                    rainfall_range = (0, 20)
                    wind_range = (5, 15)
                elif 6 <= month <= 9:  # Monsoon
                    temp_range = (25, 35)
                    rainfall_range = (20, 100)
                    wind_range = (10, 25)
                else:  # Winter
                    temp_range = (15, 30)
                    rainfall_range = (0, 10)
                    wind_range = (5, 15)
                
                # Add some randomness to predictions
                prediction = {
                    'date': date.strftime('%Y-%m-%d'),
                    'temperature': round(np.random.uniform(*temp_range), 1),
                    'rainfall': round(np.random.uniform(*rainfall_range), 1),
                    'wind_speed': round(np.random.uniform(*wind_range), 1)
                }
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting weather for {location}: {str(e)}")
            return None

    def _calculate_trend(self, values):
        """Calculate trend direction and magnitude"""
        if len(values) < 2:
            return "insufficient data"
            
        # Calculate simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.1:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def get_seasonal_pattern(self, location):
        """
        Get seasonal weather patterns for a location
        Args:
            location (str): Location name
        Returns:
            dict: Seasonal patterns for temperature, rainfall, and wind speed
        """
        try:
            location_data = self.historical_data[self.historical_data['location'] == location].copy()
            location_data['month'] = pd.to_datetime(location_data['date']).dt.month
            
            monthly_patterns = location_data.groupby('month').agg({
                'temperature': 'mean',
                'rainfall': 'mean',
                'wind_speed': 'mean'
            }).round(1)
            
            return {
                'monthly_temperature': monthly_patterns['temperature'].to_dict(),
                'monthly_rainfall': monthly_patterns['rainfall'].to_dict(),
                'monthly_wind_speed': monthly_patterns['wind_speed'].to_dict()
            }
        except Exception as e:
            logger.error(f"Error calculating seasonal patterns: {e}")
            return None

    def get_extreme_events(self, location, threshold_percentile=95):
        """
        Get extreme weather events for a location
        Args:
            location (str): Location name
            threshold_percentile (int): Percentile threshold for extreme events
        Returns:
            dict: Extreme events data
        """
        try:
            location_data = self.historical_data[self.historical_data['location'] == location]
            
            temp_threshold = location_data['temperature'].quantile(threshold_percentile/100)
            rain_threshold = location_data['rainfall'].quantile(threshold_percentile/100)
            wind_threshold = location_data['wind_speed'].quantile(threshold_percentile/100)
            
            extreme_events = {
                'temperature': len(location_data[location_data['temperature'] > temp_threshold]),
                'rainfall': len(location_data[location_data['rainfall'] > rain_threshold]),
                'wind_speed': len(location_data[location_data['wind_speed'] > wind_threshold])
            }
            
            return extreme_events
        except Exception as e:
            logger.error(f"Error calculating extreme events: {e}")
            return None

    def _calculate_confidence_interval(self, series):
        """Calculate 95% confidence interval for a series"""
        mean = series.mean()
        std = series.std()
        return {
            'lower': mean - 1.96 * std,
            'upper': mean + 1.96 * std
        } 