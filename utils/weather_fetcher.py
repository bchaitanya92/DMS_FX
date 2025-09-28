import requests
import os
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WeatherDataFetcher:
    def __init__(self):
        """Initialize WeatherDataFetcher with OpenWeather API"""
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenWeather API key not found in environment variables")
            
        self.locations = {
            'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
            'Delhi': {'lat': 28.6139, 'lon': 77.2090},
            'Chennai': {'lat': 13.0827, 'lon': 80.2707},
            'Kolkata': {'lat': 22.5726, 'lon': 88.3639},
            'Bangalore': {'lat': 12.9716, 'lon': 77.5946}
        }
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
    def get_all_locations(self):
        """Return list of all available locations"""
        return list(self.locations.keys())

    def get_weather(self, location):
        """Get real-time weather data for a location using OpenWeather API"""
        if location not in self.locations:
            raise ValueError(f"Location {location} not found")
            
        coords = self.locations[location]
        
        try:
            # Get current weather data
            current_url = f"{self.base_url}/weather"
            params = {
                'lat': coords['lat'],
                'lon': coords['lon'],
                'appid': self.api_key,
                'units': 'metric'  # Use metric units
            }
            
            response = requests.get(current_url, params=params)
            response.raise_for_status()
            weather_data = response.json()
            
            # Extract relevant weather parameters
            return {
                'lat': coords['lat'],
                'lon': coords['lon'],
                'temperature': weather_data['main']['temp'],
                'wind_speed': weather_data['wind']['speed'] * 3.6,  # Convert m/s to km/h
                'rainfall': weather_data.get('rain', {}).get('1h', 0),  # Rain in last hour (mm)
                'humidity': weather_data['main']['humidity'],
                'pressure': weather_data['main']['pressure'],
                'weather_condition': weather_data['weather'][0]['main'],
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.RequestException as e:
            logging.error(f"Error fetching weather data: {str(e)}")
            # Return None or raise exception based on your error handling preference
            raise

    def get_historical_data(self, location, days=5):
        """Get historical weather data for training"""
        if location not in self.locations:
            raise ValueError(f"Location {location} not found")
            
        coords = self.locations[location]
        
        try:
            # Get 5-day forecast data (includes historical data points)
            forecast_url = f"{self.base_url}/forecast"
            params = {
                'lat': coords['lat'],
                'lon': coords['lon'],
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(forecast_url, params=params)
            response.raise_for_status()
            forecast_data = response.json()
            
            # Process and format the data
            historical_data = []
            for item in forecast_data['list']:
                historical_data.append({
                    'timestamp': datetime.fromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'wind_speed': item['wind']['speed'] * 3.6,  # Convert m/s to km/h
                    'rainfall': item.get('rain', {}).get('3h', 0) / 3,  # Convert 3h rain to 1h average
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'weather_condition': item['weather'][0]['main']
                })
            
            return historical_data
            
        except requests.RequestException as e:
            logging.error(f"Error fetching historical data: {str(e)}")
            raise 