import logging
from utils.weather_fetcher import WeatherDataFetcher
from utils.data_collector import DisasterDataCollector
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_collection():
    try:
        # Initialize components
        logger.info("Initializing components...")
        weather_fetcher = WeatherDataFetcher()
        data_collector = DisasterDataCollector(weather_fetcher)
        
        # Test location
        test_location = "Mumbai"
        
        # 1. Test Weather Data
        logger.info(f"\nTesting weather data collection for {test_location}...")
        weather_data = weather_fetcher.get_weather(test_location)
        print("\nWeather Data:")
        print(json.dumps(weather_data, indent=2))
        
        # 2. Test Complete Data Collection
        logger.info(f"\nTesting complete data collection for {test_location}...")
        complete_data = data_collector.collect_real_time_data(test_location)
        print("\nComplete Data:")
        print(json.dumps(complete_data, indent=2))
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_data_collection() 