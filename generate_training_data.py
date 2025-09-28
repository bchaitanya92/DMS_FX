import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_historical_data(start_date, end_date, locations):
    """Generate synthetic weather and disaster data for training"""
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    data = []
    
    # Base weather patterns for seasons
    seasons = {
        'winter': {'temp': (5, 15), 'wind': (5, 20), 'rainfall': (0, 10)},
        'spring': {'temp': (15, 25), 'wind': (10, 25), 'rainfall': (5, 20)},
        'summer': {'temp': (25, 35), 'wind': (5, 15), 'rainfall': (0, 5)},
        'monsoon': {'temp': (20, 30), 'wind': (15, 35), 'rainfall': (20, 100)}
    }
    
    # Disaster probability increases with extreme weather conditions
    for date in dates:
        month = date.month
        # Determine season
        if month in [12, 1, 2]:
            season = 'winter'
        elif month in [3, 4, 5]:
            season = 'spring'
        elif month in [6, 7, 8]:
            season = 'summer'
        else:
            season = 'monsoon'
            
        for location in locations:
            # Base weather for the season
            base = seasons[season]
            
            # Add random variations
            temperature = np.random.uniform(base['temp'][0], base['temp'][1])
            wind_speed = np.random.uniform(base['wind'][0], base['wind'][1])
            rainfall = np.random.uniform(base['rainfall'][0], base['rainfall'][1])
            
            # Add occasional extreme weather events
            if random.random() < 0.05:  # 5% chance of extreme weather
                if season == 'monsoon':
                    rainfall *= 2.5  # Heavy rainfall
                    wind_speed *= 1.5  # Strong winds
                elif season == 'summer':
                    temperature *= 1.3  # Heat wave
                    
            # Determine if a disaster occurred based on weather conditions
            disaster_type = 'none'
            if rainfall > 80:
                disaster_type = 'flood'
            elif wind_speed > 45:
                disaster_type = 'storm'
            elif temperature > 45:
                disaster_type = 'fire'
            elif random.random() < 0.001:  # Random earthquakes (0.1% chance)
                disaster_type = 'earthquake'
            
            data.append({
                'timestamp': date,
                'location': location,
                'temperature': round(temperature, 2),
                'wind_speed': round(wind_speed, 2),
                'rainfall': round(rainfall, 2),
                'humidity': round(random.uniform(60, 90), 2),
                'pressure': round(random.uniform(980, 1020), 2),
                'disaster_type': disaster_type
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate 2 years of data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    locations = [
        'Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Bangalore',
        'Hyderabad', 'Ahmedabad', 'Pune', 'Jaipur', 'Surat'
    ]
    
    print("Generating 2 years of historical data...")
    df = generate_historical_data(start_date, end_date, locations)
    
    # Save to CSV
    print("Saving data to historical_weather_data.csv...")
    df.to_csv('data/historical_weather_data.csv', index=False)
    print(f"Generated {len(df)} records of weather and disaster data")
    
    # Print some statistics
    print("\nDisaster Statistics:")
    print(df['disaster_type'].value_counts())
    
    print("\nWeather Statistics:")
    for col in ['temperature', 'wind_speed', 'rainfall']:
        print(f"\n{col.title()}:")
        print(f"Mean: {df[col].mean():.2f}")
        print(f"Max: {df[col].max():.2f}")
        print(f"Min: {df[col].min():.2f}") 