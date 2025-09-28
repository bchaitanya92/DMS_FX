import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_training_data(n_samples=1000):
    """Generate training data with various disaster scenarios"""
    logger.info(f"Generating {n_samples} training samples")
    
    # Generate random environmental conditions
    data = {
        'rainfall': np.random.uniform(0, 500, n_samples),
        'temperature': np.random.uniform(25, 275, n_samples),
        'seismic_activity': np.random.uniform(0, 10, n_samples),
        'wind_speed': np.random.uniform(10, 510, n_samples)
    }
    
    # Initialize disaster types (0: No Disaster, 1: Flood, 2: Earthquake, 3: Storm)
    disaster_types = np.zeros(n_samples)
    
    # Define disaster conditions
    flood_conditions = (
        (data['rainfall'] > 300) &  # High rainfall
        (data['temperature'] > 150)  # High temperature
    )
    
    earthquake_conditions = (
        (data['seismic_activity'] > 7) &  # High seismic activity
        (data['temperature'] < 100)  # Lower temperature
    )
    
    storm_conditions = (
        (data['wind_speed'] > 400) &  # High wind speed
        (data['rainfall'] > 200)  # Moderate rainfall
    )
    
    # Assign disaster types based on conditions
    disaster_types[flood_conditions] = 1
    disaster_types[earthquake_conditions] = 2
    disaster_types[storm_conditions] = 3
    
    # Add disaster types to data
    data['disaster_type'] = disaster_types
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = 'training_data.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"Training data saved to {output_file}")
    
    # Print summary
    disaster_counts = df['disaster_type'].value_counts()
    logger.info("\nDisaster Type Distribution:")
    for disaster_type, count in disaster_counts.items():
        disaster_name = {
            0: 'No Disaster',
            1: 'Flood',
            2: 'Earthquake',
            3: 'Storm'
        }[disaster_type]
        logger.info(f"{disaster_name}: {count} samples")
    
    return df

if __name__ == '__main__':
    generate_training_data() 