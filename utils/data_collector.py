import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
import os
from sklearn.preprocessing import StandardScaler
import json
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class DisasterDataCollector:
    def __init__(self, weather_fetcher):
        """Initialize the DisasterDataCollector with a weather fetcher instance"""
        self.weather_fetcher = weather_fetcher
        self.scaler = StandardScaler()
        self.historical_data_file = 'data/historical_disaster_data.csv'
        self.feature_columns = [
            'temperature', 'wind_speed', 'rainfall', 'humidity', 
            'pressure', 'population_density', 'elevation',
            'distance_to_water', 'soil_moisture'
        ]
        
        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('cache', exist_ok=True)
        
        # Initialize API endpoints
        self.air_quality_url = "https://api.waqi.info/feed"  # World Air Quality Index (free tier)
        self.elevation_url = "https://api.open-elevation.com/api/v1/lookup"  # Open Elevation API
        self.imd_public_url = "https://internal.imd.gov.in/pages/city_weather_main_mausam.php"  # IMD Public Data
        
        # Load or create geographical data
        self.geo_data = self._load_geographical_data()
        
    def collect_real_time_data(self, location):
        """Collect real-time data from free sources"""
        try:
            # Get weather data
            weather_data = self.weather_fetcher.get_weather(location)
            lat, lon = weather_data['lat'], weather_data['lon']
            
            # Get air quality data
            air_quality = self._fetch_air_quality_data(lat, lon)
            
            # Get elevation data
            elevation_data = self._fetch_elevation_data(lat, lon)
            
            # Get IMD data
            imd_data = self._fetch_imd_public_data(location)
            
            # Get geographical data
            geo_data = self.geo_data.get(location, {})
            
            # Combine all data
            real_time_data = {
                'timestamp': datetime.now().isoformat(),
                'location': location,
                'coordinates': {'lat': lat, 'lon': lon},
                'weather': weather_data,
                'air_quality': air_quality if air_quality else {},
                'elevation': elevation_data if elevation_data else {},
                'imd_weather': imd_data if imd_data else {},
                'geographical': geo_data,
                'update_frequency': '15 minutes'
            }
            
            # Calculate disaster probabilities
            probabilities = self._calculate_disaster_probabilities(real_time_data)
            real_time_data['disaster_probabilities'] = probabilities
            
            # Save to cache
            cache_file = f"cache/real_time_data_{location.lower().replace(' ', '_')}.json"
            with open(cache_file, 'w') as f:
                json.dump(real_time_data, f)
            
            return real_time_data
            
        except Exception as e:
            logger.error(f"Error collecting real-time data: {str(e)}")
            return None
            
    def _fetch_air_quality_data(self, lat, lon):
        """Fetch air quality data from WAQI (free tier)"""
        try:
            response = requests.get(f"{self.air_quality_url}/geo:{lat};{lon}/?token=demo")
            if response.status_code == 200:
                data = response.json()
                return {
                    'aqi': data.get('data', {}).get('aqi', 0),
                    'pm25': data.get('data', {}).get('iaqi', {}).get('pm25', {}).get('v', 0),
                    'pm10': data.get('data', {}).get('iaqi', {}).get('pm10', {}).get('v', 0)
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching air quality data: {str(e)}")
            return None

    def _fetch_elevation_data(self, lat, lon):
        """Fetch elevation data from Open Elevation API"""
        try:
            payload = {
                "locations": [
                    {"latitude": lat, "longitude": lon}
                ]
            }
            response = requests.post(self.elevation_url, json=payload)
            if response.status_code == 200:
                data = response.json()
                elevation = data['results'][0]['elevation']
                return {
                    'elevation': elevation,
                    'slope': self._calculate_slope(elevation)
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching elevation data: {str(e)}")
            return None

    def _fetch_imd_public_data(self, location):
        """Fetch public weather data from IMD website"""
        try:
            response = requests.get(self.imd_public_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                weather_table = soup.find('table', {'class': 'tableData'})
                
                if weather_table:
                    rows = weather_table.find_all('tr')
                    for row in rows[1:]:  # Skip header row
                        cols = row.find_all('td')
                        if len(cols) >= 6 and cols[0].text.strip() == location:
                            return {
                                'temperature': float(cols[1].text.strip()),
                                'humidity': float(cols[2].text.strip()),
                                'rainfall': float(cols[3].text.strip()),
                                'wind_speed': float(cols[4].text.strip()),
                                'wind_direction': cols[5].text.strip()
                            }
            return None
        except Exception as e:
            logger.error(f"Error fetching IMD public data: {str(e)}")
            return None

    def _calculate_slope(self, elevation, radius=0.01):
        """Calculate approximate slope using elevation"""
        try:
            return min((elevation / (radius * 1000)) * 100, 100)
        except:
            return 0

    def _calculate_disaster_probabilities(self, data):
        """Calculate disaster probabilities based on current conditions"""
        probabilities = {
            'flood': 0.0,
            'cyclone': 0.0,
            'heatwave': 0.0,
            'landslide': 0.0
        }
        
        # Get weather data
        weather = data['weather']
        elevation = data.get('elevation', {}).get('elevation', 0)
        slope = data.get('elevation', {}).get('slope', 0)
        
        # Calculate flood probability
        if weather['rainfall'] > 50:
            probabilities['flood'] = min(weather['rainfall'] / 200, 0.9)
            if elevation < 10:  # Low-lying area
                probabilities['flood'] += 0.1
                
        # Calculate cyclone probability
        if weather['wind_speed'] > 40:
            probabilities['cyclone'] = min(weather['wind_speed'] / 100, 0.9)
            if data.get('geographical', {}).get('region_type') == 'coastal':
                probabilities['cyclone'] += 0.1
                
        # Calculate heatwave probability
        if weather['temperature'] > 40:
            probabilities['heatwave'] = min((weather['temperature'] - 40) / 10, 0.9)
            if weather['humidity'] > 60:
                probabilities['heatwave'] += 0.1
                
        # Calculate landslide probability
        if slope > 30 and weather['rainfall'] > 30:
            probabilities['landslide'] = min((slope / 100) * (weather['rainfall'] / 100), 0.9)
            
        return probabilities

    def _load_geographical_data(self):
        """Load or create geographical data for Indian cities"""
        geo_data = {
            'Mumbai': {
                'elevation': 14,
                'population_density': 21000,
                'distance_to_water': 0,
                'soil_type': 'coastal alluvial',
                'region_type': 'coastal'
            },
            'Delhi': {
                'elevation': 216,
                'population_density': 11320,
                'distance_to_water': 5,
                'soil_type': 'alluvial',
                'region_type': 'plains'
            },
            'Chennai': {
                'elevation': 6,
                'population_density': 26903,
                'distance_to_water': 0,
                'soil_type': 'coastal alluvial',
                'region_type': 'coastal'
            },
            'Kolkata': {
                'elevation': 9,
                'population_density': 24000,
                'distance_to_water': 0,
                'soil_type': 'alluvial',
                'region_type': 'coastal'
            }
        }
        return geo_data

    def _load_historical_imd_data(self):
        """
        Load historical weather data from IMD
        Using sample data structure, replace with actual IMD API integration
        """
        try:
            # If historical data file exists, load it
            if os.path.exists('data/imd_historical.csv'):
                return pd.read_csv('data/imd_historical.csv')
            
            # Generate sample historical data if file doesn't exist
            dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
            locations = list(self.geo_data.keys())
            data = []
            
            for location in locations:
                for date in dates:
                    # Generate realistic seasonal weather patterns
                    month = date.month
                    is_monsoon = 6 <= month <= 9
                    is_summer = 3 <= month <= 5
                    
                    base_temp = 25 + (10 if is_summer else 0)
                    base_rain = 50 if is_monsoon else 5
                    base_humidity = 80 if is_monsoon else (50 if is_summer else 65)
                    
                    # Add random variations
                    temp = base_temp + np.random.normal(0, 3)
                    rain = max(0, np.random.exponential(base_rain))
                    humidity = min(100, max(30, base_humidity + np.random.normal(0, 10)))
                    
                    data.append({
                        'date': date,
                        'location': location,
                        'temperature': temp,
                        'rainfall': rain,
                        'humidity': humidity,
                        'pressure': np.random.normal(1013, 5),
                        'wind_speed': np.random.exponential(20),
                        'soil_moisture': min(100, max(0, humidity - 20 + np.random.normal(0, 5)))
                    })
            
            df = pd.DataFrame(data)
            df.to_csv('data/imd_historical.csv', index=False)
            return df
            
        except Exception as e:
            logging.error(f"Error loading historical IMD data: {str(e)}")
            return pd.DataFrame()

    def _load_disaster_records(self):
        """
        Load historical disaster records
        Using structured sample data based on real patterns
        """
        try:
            if os.path.exists('data/disaster_records.csv'):
                return pd.read_csv('data/disaster_records.csv')
            
            # Generate realistic disaster records
            data = []
            locations = list(self.geo_data.keys())
            
            # Define disaster patterns
            disaster_patterns = {
                'Mumbai': {
                    'flood': {'probability': 0.4, 'monsoon_boost': 0.4},
                    'cyclone': {'probability': 0.2, 'monsoon_boost': 0.2},
                    'heatwave': {'probability': 0.1, 'summer_boost': 0.3}
                },
                'Delhi': {
                    'flood': {'probability': 0.2, 'monsoon_boost': 0.3},
                    'heatwave': {'probability': 0.4, 'summer_boost': 0.4},
                    'storm': {'probability': 0.2, 'monsoon_boost': 0.2}
                },
                'Chennai': {
                    'flood': {'probability': 0.4, 'monsoon_boost': 0.4},
                    'cyclone': {'probability': 0.3, 'monsoon_boost': 0.3},
                    'heatwave': {'probability': 0.2, 'summer_boost': 0.3}
                },
                'Kolkata': {
                    'flood': {'probability': 0.3, 'monsoon_boost': 0.4},
                    'cyclone': {'probability': 0.3, 'monsoon_boost': 0.3},
                    'heatwave': {'probability': 0.2, 'summer_boost': 0.3}
                },
                'Bangalore': {
                    'flood': {'probability': 0.2, 'monsoon_boost': 0.3},
                    'heatwave': {'probability': 0.2, 'summer_boost': 0.3},
                    'storm': {'probability': 0.1, 'monsoon_boost': 0.2}
                }
            }
            
            # Generate data for past 4 years
            dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
            
            for location in locations:
                patterns = disaster_patterns[location]
                
                for date in dates:
                    month = date.month
                    is_monsoon = 6 <= month <= 9
                    is_summer = 3 <= month <= 5
                    
                    for disaster_type, pattern in patterns.items():
                        base_prob = pattern['probability']
                        if is_monsoon and 'monsoon_boost' in pattern:
                            base_prob += pattern['monsoon_boost']
                        if is_summer and 'summer_boost' in pattern:
                            base_prob += pattern['summer_boost']
                            
                        if np.random.random() < base_prob:
                            severity = np.random.choice(['low', 'medium', 'high'], 
                                                      p=[0.5, 0.3, 0.2])
                            data.append({
                                'date': date,
                                'location': location,
                                'disaster_type': disaster_type,
                                'severity': severity,
                                'affected_population': int(np.random.exponential(1000)),
                                'economic_impact': int(np.random.exponential(1000000))
                            })
            
            df = pd.DataFrame(data)
            df.to_csv('data/disaster_records.csv', index=False)
            return df
            
        except Exception as e:
            logging.error(f"Error loading disaster records: {str(e)}")
            return pd.DataFrame()

    def collect_training_data(self):
        """Collect and prepare training data by combining weather and disaster records"""
        try:
            # Load historical weather and disaster data
            weather_data = self._load_historical_imd_data()
            disaster_data = self._load_disaster_records()
            
            if weather_data.empty or disaster_data.empty:
                raise ValueError("Failed to load required data")
            
            # Merge weather and disaster data
            df = weather_data.merge(
                disaster_data, 
                on=['date', 'location'], 
                how='left'
            )
            
            # Add geographical features
            df['elevation'] = df['location'].map({k: v['elevation'] 
                                                for k, v in self.geo_data.items()})
            df['population_density'] = df['location'].map({k: v['population_density'] 
                                                         for k, v in self.geo_data.items()})
            df['distance_to_water'] = df['location'].map({k: v['distance_to_water'] 
                                                        for k, v in self.geo_data.items()})
            
            # Create target variables
            df['has_disaster'] = df['disaster_type'].notna().astype(int)
            df['is_flood'] = (df['disaster_type'] == 'flood').astype(int)
            df['is_cyclone'] = (df['disaster_type'] == 'cyclone').astype(int)
            df['is_heatwave'] = (df['disaster_type'] == 'heatwave').astype(int)
            df['is_storm'] = (df['disaster_type'] == 'storm').astype(int)
            
            # Calculate additional features
            df['temp_humidity_index'] = df['temperature'] * df['humidity'] / 100
            df['rain_wind_index'] = df['rainfall'] * df['wind_speed'] / 100
            
            # Save processed data
            df.to_csv(self.historical_data_file, index=False)
            logging.info(f"Training data saved to {self.historical_data_file}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error collecting training data: {str(e)}")
            return None

    def get_current_conditions(self, location):
        """Get current weather and geographical conditions for a location"""
        try:
            # Get real-time weather data
            weather_data = self.weather_fetcher.get_weather(location)
            
            # Combine with geographical data
            conditions = {
                **weather_data,
                **self.geo_data[location]
            }
            
            # Calculate derived features
            conditions['temp_humidity_index'] = (
                conditions['temperature'] * conditions['humidity'] / 100
            )
            conditions['rain_wind_index'] = (
                conditions['rainfall'] * conditions['wind_speed'] / 100
            )
            
            return conditions
            
        except Exception as e:
            logging.error(f"Error getting current conditions: {str(e)}")
            return None

    def get_feature_importance(self):
        """Get feature importance based on historical data"""
        try:
            df = pd.read_csv(self.historical_data_file)
            
            # Calculate correlation with disaster occurrence
            correlations = {}
            for feature in self.feature_columns:
                if feature in df.columns:
                    corr = df[feature].corr(df['has_disaster'])
                    correlations[feature] = abs(corr)
            
            return correlations
            
        except Exception as e:
            logging.error(f"Error calculating feature importance: {str(e)}")
            return {}

    def collect_historical_disaster_data(self):
        """
        Collect historical disaster data from reliable sources and combine with weather data
        """
        # Initialize empty list to store all data
        all_data = []
        
        # Get data for each location
        for location in self.weather_fetcher.get_all_locations():
            try:
                # Get historical weather data
                weather_data = self.weather_fetcher.get_historical_data(location)
                
                for data_point in weather_data:
                    # Define disaster probability based on weather conditions
                    disaster_probs = self._calculate_disaster_probabilities(data_point)
                    
                    # Add location and disaster probabilities to weather data
                    entry = {
                        'location': location,
                        'timestamp': data_point['timestamp'],
                        'temperature': data_point['temperature'],
                        'wind_speed': data_point['wind_speed'],
                        'rainfall': data_point['rainfall'],
                        'humidity': data_point['humidity'],
                        'pressure': data_point['pressure'],
                        'weather_condition': data_point['weather_condition']
                    }
                    
                    # Add disaster probabilities
                    entry.update(disaster_probs)
                    all_data.append(entry)
                    
            except Exception as e:
                logging.error(f"Error collecting data for {location}: {str(e)}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        return df
    
    def save_to_csv(self, df, filename='data/historical_disaster_data.csv'):
        """Save the collected data to CSV file"""
        df.to_csv(filename, index=False)
        logging.info(f"Data saved to {filename}")
        
    def load_from_csv(self, filename='data/historical_disaster_data.csv'):
        """Load historical data from CSV file"""
        try:
            return pd.read_csv(filename)
        except FileNotFoundError:
            logging.error(f"File {filename} not found")
            return None 