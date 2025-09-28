import requests
import logging
from datetime import datetime, timedelta
import math
import random
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class WeatherDataFetcher:
    """Class to fetch real-time weather data"""
    
    def __init__(self):
        """Initialize the WeatherDataFetcher"""
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.locations = {
            'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
            'Delhi': {'lat': 28.6139, 'lon': 77.2090},
            'Chennai': {'lat': 13.0827, 'lon': 80.2707},
            'Kolkata': {'lat': 22.5726, 'lon': 88.3639},
            'Bangalore': {'lat': 12.9716, 'lon': 77.5946},
            'Hyderabad': {'lat': 17.3850, 'lon': 78.4867},
            'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714},
            'Pune': {'lat': 18.5204, 'lon': 73.8567},
            'Jaipur': {'lat': 26.9124, 'lon': 75.7873},
            'Surat': {'lat': 21.1702, 'lon': 72.8311}
        }
        logger.info("WeatherDataFetcher initialized successfully")

    def get_weather(self, location):
        """Get current weather data for a location"""
        if location not in self.locations:
            raise ValueError(f"Location {location} not found")
        
        try:
            if not self.api_key:
                logger.warning("OpenWeather API key not found, using simulated data")
                return self._get_simulated_weather(location)
            
            # Make API request using coordinates
            params = {
                'lat': self.locations[location]['lat'],
                'lon': self.locations[location]['lon'],
                'appid': self.api_key,
                'units': 'metric'  # Use metric units
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract weather data
            weather_data = {
                'temperature': data['main']['temp'],
                'wind_speed': data['wind']['speed'] * 3.6,  # Convert m/s to km/h
                'rainfall': data.get('rain', {}).get('1h', 0),  # Rainfall in last hour
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'location': location,
                'lat': self.locations[location]['lat'],
                'lon': self.locations[location]['lon'],
                'timestamp': datetime.now().isoformat(),
                'description': data['weather'][0]['description'],
                'clouds': data['clouds']['all']
            }
            
            logger.info(f"Successfully fetched weather data for {location}")
            return weather_data
            
        except Exception as e:
            logger.error(f"Error fetching weather data for {location}: {str(e)}")
            logger.info("Falling back to simulated data")
            return self._get_simulated_weather(location)

    def _get_simulated_weather(self, location):
        """Generate simulated weather data based on season"""
        current_month = datetime.now().month
        
        # Define seasonal variations
        if 3 <= current_month <= 5:  # Summer
            temperature = np.random.normal(35, 5)
            rainfall = np.random.normal(10, 5)
            wind_speed = np.random.normal(8, 2)
            humidity = np.random.normal(50, 10)
        elif 6 <= current_month <= 9:  # Monsoon
            temperature = np.random.normal(30, 3)
            rainfall = np.random.normal(100, 30)
            wind_speed = np.random.normal(15, 5)
            humidity = np.random.normal(80, 10)
        else:  # Winter
            temperature = np.random.normal(25, 5)
            rainfall = np.random.normal(5, 2)
            wind_speed = np.random.normal(5, 2)
            humidity = np.random.normal(60, 10)
            
        # Ensure values are within realistic ranges
        temperature = max(15, min(45, temperature))
        rainfall = max(0, rainfall)
        wind_speed = max(0, min(50, wind_speed))
        humidity = max(0, min(100, humidity))
        
        return {
            'temperature': round(temperature, 1),
            'wind_speed': round(wind_speed, 1),
            'rainfall': round(rainfall, 1),
            'humidity': round(humidity, 1),
            'pressure': round(np.random.normal(1013, 5), 1),
            'location': location,
            'lat': self.locations[location]['lat'],
            'lon': self.locations[location]['lon'],
            'timestamp': datetime.now().isoformat(),
            'description': 'simulated data',
            'clouds': round(np.random.normal(50, 20))
        }

    def get_forecast(self, location, days=7):
        """Get weather forecast for a location"""
        if location not in self.locations:
            raise ValueError(f"Location {location} not found")
            
        forecast = []
        current_month = datetime.now().month
        
        for day in range(days):
            date = datetime.now() + timedelta(days=day)
            
            # Generate weather data based on season
            if 3 <= current_month <= 5:  # Summer
                temperature = np.random.normal(35, 5)
                rainfall = np.random.normal(10, 5)
                wind_speed = np.random.normal(8, 2)
            elif 6 <= current_month <= 9:  # Monsoon
                temperature = np.random.normal(30, 3)
                rainfall = np.random.normal(100, 30)
                wind_speed = np.random.normal(15, 5)
            else:  # Winter
                temperature = np.random.normal(25, 5)
                rainfall = np.random.normal(5, 2)
                wind_speed = np.random.normal(5, 2)
            
            # Ensure values are within realistic ranges
            temperature = max(15, min(45, temperature))
            rainfall = max(0, rainfall)
            wind_speed = max(0, min(50, wind_speed))
            
            forecast.append({
                'date': date.isoformat(),
                'temperature': round(temperature, 1),
                'wind_speed': round(wind_speed, 1),
                'rainfall': round(rainfall, 1)
            })
            
        return forecast

    def get_historical_data(self, location, days=30):
        """
        Get historical weather data for a location
        Args:
            location (str): Name of the location
            days (int): Number of past days to get data for
        Returns:
            list: List of daily historical weather data
        """
        if location not in self.locations:
            raise ValueError(f"Location {location} not found")

        historical_data = []
        current_weather = self.get_weather(location)

        for day in range(days, 0, -1):
            # Generate historical data based on current weather with some variation
            date = datetime.now() - timedelta(days=day)
            seasonal_factor = math.sin((date.timetuple().tm_yday / 365) * 2 * math.pi)
            
            historical_day = {
                'date': date.date().isoformat(),
                'temperature': round(current_weather['temperature'] + seasonal_factor * 5 + random.uniform(-3, 3), 1),
                'wind_speed': round(max(0, current_weather['wind_speed'] + random.uniform(-5, 5)), 1),
                'rainfall': round(max(0, current_weather['rainfall'] + random.uniform(-2, 2)), 1),
                'humidity': round(min(100, max(0, current_weather['humidity'] + random.uniform(-10, 10))), 1)
            }
            historical_data.append(historical_day)

        return historical_data

    def get_weather_data(self, location):
        """Fetch real-time weather data for a given location"""
        try:
            # Make API request
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric'  # Use metric units
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise exception for bad status codes
            
            data = response.json()
            
            # Extract relevant weather data
            weather_data = {
                'Temperature': data['main']['temp'],
                'Wind Speed': data['wind']['speed'] * 3.6,  # Convert m/s to km/h
                'Rainfall': data.get('rain', {}).get('1h', 0),  # Rainfall in last hour
                'Humidity': data['main']['humidity'],
                'Pressure': data['main']['pressure'],
                'Timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully fetched weather data for {location}")
            return weather_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data for {location}: {str(e)}")
            return None
        except KeyError as e:
            logger.error(f"Error parsing weather data for {location}: {str(e)}")
            return None

    def get_all_locations(self):
        """Get all available locations"""
        # Return a list of locations instead of a dictionary
        return [
            'Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Bangalore',
            'Hyderabad', 'Ahmedabad', 'Pune', 'Jaipur', 'Surat'
        ] 