import pandas as pd
import numpy as np
import requests
import logging
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import json
import netCDF4
from bs4 import BeautifulSoup
import googlemaps
from sentinelsat import SentinelAPI
from PIL import Image
import io
import ee
import time

logger = logging.getLogger(__name__)

class DisasterDataCollector:
    def __init__(self, weather_fetcher):
        """Initialize the DisasterDataCollector with a weather fetcher instance"""
        self.weather_fetcher = weather_fetcher
        self.scaler = StandardScaler()
        self.feature_columns = [
            # Weather features
            'temperature', 'wind_speed', 'rainfall', 'humidity', 'pressure',
            'visibility', 'cloud_cover', 'wind_direction',
            
            # Environmental features
            'soil_moisture', 'air_quality_index', 'river_level',
            'vegetation_index',
            
            # Geographical features
            'elevation', 'slope', 'population_density', 
            'distance_to_water', 'land_use_index',
            
            # Infrastructure features
            'building_density', 'road_density',
            'emergency_services_proximity',
            
            # Historical disaster features
            'historical_flood_frequency', 'historical_cyclone_frequency',
            'historical_heatwave_frequency'
        ]
        
        # Initialize Google Earth Engine
        try:
            ee.Initialize()
            self.ee_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Google Earth Engine: {str(e)}")
            self.ee_initialized = False
        
        # Load API keys
        self.google_maps_key = os.getenv('GOOGLE_MAPS_API_KEY', '')
        self.tomtom_key = os.getenv('TOMTOM_API_KEY', '')
        self.sentinel_user = os.getenv('SENTINEL_USER', '')
        self.sentinel_password = os.getenv('SENTINEL_PASSWORD', '')
        
        # Initialize APIs
        self.gmaps = googlemaps.Client(key=self.google_maps_key) if self.google_maps_key else None
        self.sentinel_api = SentinelAPI(self.sentinel_user, self.sentinel_password, 'https://scihub.copernicus.eu/dhus')
        
        # Create cache directory
        os.makedirs('cache', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Initialize data sources
        self._initialize_data_sources()

    def _initialize_data_sources(self):
        """Initialize connections to various data sources"""
        # Base URLs for free APIs
        self.openstreetmap_url = "https://www.openstreetmap.org/api/0.6"
        self.air_quality_url = "https://api.waqi.info/feed"  # World Air Quality Index (free tier)
        self.elevation_url = "https://api.open-elevation.com/api/v1/lookup"  # Open Elevation API
        self.imd_public_url = "https://internal.imd.gov.in/pages/city_weather_main_mausam.php"  # IMD Public Data
        
    def _fetch_osm_infrastructure_data(self, location):
        """Fetch infrastructure data from OpenStreetMap"""
        try:
            # Use OSM Overpass API (free)
            overpass_url = "https://overpass-api.de/api/interpreter"
            query = f"""
            [out:json][timeout:25];
            area[name="{location}"]->.searchArea;
            (
              way["highway"](area.searchArea);
              way["building"](area.searchArea);
              way["amenity"="hospital"](area.searchArea);
              way["amenity"="fire_station"](area.searchArea);
              way["amenity"="police"](area.searchArea);
            );
            out body;
            >;
            out skel qt;
            """
            
            response = requests.get(overpass_url, params={'data': query})
            
            if response.status_code == 200:
                data = response.json()
                
                # Calculate infrastructure metrics
                building_count = len([e for e in data['elements'] if 'building' in e.get('tags', {})])
                road_length = sum(self._calculate_way_length(e) for e in data['elements'] if 'highway' in e.get('tags', {}))
                emergency_services = len([e for e in data['elements'] 
                                       if e.get('tags', {}).get('amenity') in ['hospital', 'fire_station', 'police']])
                
                area = self._calculate_area(data['elements'])
                
                return {
                    'building_density': building_count / area if area > 0 else 0,
                    'road_density': road_length / area if area > 0 else 0,
                    'emergency_services_density': emergency_services / area if area > 0 else 0
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching OSM data: {str(e)}")
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
                return {
                    'elevation': data['results'][0]['elevation'],
                    'slope': self._calculate_slope(data['results'][0]['elevation'])
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching elevation data: {str(e)}")
            return None

    def _fetch_imd_public_data(self, location):
        """Fetch public weather data from IMD website"""
        try:
            # Scrape IMD public website
            response = requests.get(self.imd_public_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Find the table containing weather data
                weather_table = soup.find('table', {'class': 'tableData'})
                
                if weather_table:
                    data = []
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

    def collect_real_time_data(self, location):
        """Collect real-time data from free sources"""
        try:
            # Get location coordinates from weather fetcher
            weather_data = self.weather_fetcher.get_weather(location)
            lat, lon = weather_data.get('lat'), weather_data.get('lon')
            
            # Fetch data from free sources
            infrastructure_data = self._fetch_osm_infrastructure_data(location)
            air_quality = self._fetch_air_quality_data(lat, lon)
            elevation_data = self._fetch_elevation_data(lat, lon)
            imd_data = self._fetch_imd_public_data(location)
            
            # Combine all real-time data
            real_time_data = {
                'timestamp': datetime.now().isoformat(),
                'location': location,
                'coordinates': {'lat': lat, 'lon': lon},
                'weather': weather_data,
                'infrastructure': infrastructure_data if infrastructure_data else {},
                'air_quality': air_quality if air_quality else {},
                'elevation': elevation_data if elevation_data else {},
                'imd_weather': imd_data if imd_data else {},
                'update_frequency': '15 minutes'
            }
            
            # Save to cache
            cache_file = f"cache/real_time_data_{location.lower().replace(' ', '_')}.json"
            with open(cache_file, 'w') as f:
                json.dump(real_time_data, f)
            
            return real_time_data
            
        except Exception as e:
            logger.error(f"Error collecting real-time data: {str(e)}")
            return None

    def _calculate_slope(self, elevation, radius=0.01):
        """Calculate approximate slope using elevation"""
        try:
            # Simple slope calculation (percentage)
            return min((elevation / (radius * 1000)) * 100, 100)
        except:
            return 0

    def _calculate_way_length(self, way):
        """Calculate length of a way in meters"""
        try:
            if 'geometry' in way:
                points = [(p['lat'], p['lon']) for p in way['geometry']]
                total_length = 0
                for i in range(len(points) - 1):
                    total_length += self._haversine_distance(points[i], points[i + 1])
                return total_length
            return 0
        except:
            return 0

    def _haversine_distance(self, point1, point2):
        """Calculate the great circle distance between two points on Earth"""
        lat1, lon1 = point1
        lat2, lon2 = point2
        R = 6371000  # Earth's radius in meters
        
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = np.sin(delta_phi/2)**2 + \
            np.cos(phi1) * np.cos(phi2) * \
            np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c

    def _calculate_area(self, elements):
        """Calculate approximate area in square kilometers"""
        try:
            # Find bounding box
            lats = []
            lons = []
            for e in elements:
                if 'lat' in e and 'lon' in e:
                    lats.append(e['lat'])
                    lons.append(e['lon'])
            
            if not lats or not lons:
                return 1  # Default 1 sq km
            
            # Calculate area using haversine formula
            width = self._haversine_distance(
                (min(lats), min(lons)),
                (min(lats), max(lons))
            )
            height = self._haversine_distance(
                (min(lats), min(lons)),
                (max(lats), min(lons))
            )
            
            return (width * height) / 1_000_000  # Convert to square kilometers
        except:
            return 1  # Default 1 sq km

    def _generate_synthetic_weather_data(self, location):
        """Generate synthetic weather data when API fails"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=365),
            end=datetime.now(),
            freq='D'
        )
        
        # Generate realistic seasonal patterns
        season_factors = {
            'summer': {'temp': 1.2, 'rain': 0.5},
            'monsoon': {'temp': 0.9, 'rain': 2.0},
            'winter': {'temp': 0.7, 'rain': 0.3}
        }
        
        data = []
        for date in dates:
            # Determine season
            month = date.month
            if month in [3, 4, 5]:
                season = 'summer'
            elif month in [6, 7, 8, 9]:
                season = 'monsoon'
            else:
                season = 'winter'
                
            factors = season_factors[season]
            
            # Generate weather data with seasonal variations
            temp_base = 25 + np.random.normal(0, 2)
            rain_base = 10 + np.random.exponential(5)
            
            data.append({
                'date': date,
                'temperature': temp_base * factors['temp'],
                'rainfall': rain_base * factors['rain'],
                'humidity': 60 + np.random.normal(0, 10),
                'wind_speed': 15 + np.random.normal(0, 5),
                'pressure': 1013 + np.random.normal(0, 2)
            })
        
        return pd.DataFrame(data)

    def _generate_synthetic_disaster_data(self):
        """Generate synthetic disaster records when API fails"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=365),
            end=datetime.now(),
            freq='D'
        )
        
        disaster_patterns = {
            'Mumbai': {
                'flood': {'probability': 0.3, 'season': [6, 7, 8, 9]},
                'cyclone': {'probability': 0.2, 'season': [5, 6, 10, 11]},
                'heatwave': {'probability': 0.15, 'season': [3, 4, 5]}
            },
            'Delhi': {
                'flood': {'probability': 0.1, 'season': [7, 8]},
                'heatwave': {'probability': 0.4, 'season': [4, 5, 6]},
                'storm': {'probability': 0.2, 'season': [6, 7]}
            }
        }
        
        data = []
        for location, patterns in disaster_patterns.items():
            for date in dates:
                for disaster_type, pattern in patterns.items():
                    if date.month in pattern['season']:
                        if np.random.random() < pattern['probability']:
                            data.append({
                                'date': date,
                                'location': location,
                                'disaster_type': disaster_type,
                                'severity': np.random.choice(['low', 'medium', 'high']),
                                'affected_population': int(np.random.exponential(1000)),
                                'economic_impact': int(np.random.exponential(1000000))
                            })
        
        return pd.DataFrame(data)

    def get_current_conditions(self, location):
        """Get current weather and environmental conditions"""
        try:
            # Get current weather
            weather_data = self.weather_fetcher.get_weather(location)
            
            # Get geographical data
            geo_data = self._load_geographical_data().get(location, {})
            
            # Combine all features
            current_conditions = {
                'temperature': weather_data.get('temperature', 0),
                'wind_speed': weather_data.get('wind_speed', 0),
                'rainfall': weather_data.get('rainfall', 0),
                'humidity': weather_data.get('humidity', 0),
                'pressure': weather_data.get('pressure', 0),
                'elevation': geo_data.get('elevation', 0),
                'population_density': geo_data.get('population_density', 0),
                'distance_to_water': geo_data.get('distance_to_water', 0),
                'land_use_index': geo_data.get('land_use_index', 0)
            }
            
            # Calculate derived features
            current_conditions['temp_humidity_index'] = (
                current_conditions['temperature'] * 1.8 + 32) - (
                0.55 - 0.0055 * current_conditions['humidity']) * (
                (current_conditions['temperature'] * 1.8 + 32) - 58)
            
            current_conditions['rain_wind_index'] = (
                current_conditions['rainfall'] * current_conditions['wind_speed']) / 100
            
            return current_conditions
            
        except Exception as e:
            logger.error(f"Error getting current conditions: {str(e)}")
            return None

    def get_feature_importance(self, location):
        """Calculate feature importance based on historical data"""
        try:
            df = pd.read_csv(f'data/training_data_{location}.csv')
            
            # Calculate correlation with disaster occurrences
            correlations = {}
            for feature in self.feature_columns:
                if feature in df.columns:
                    corr = abs(df[feature].corr(df['disaster_probability']))
                    correlations[feature] = corr
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}

    def _fetch_real_time_traffic(self, location):
        """Fetch real-time traffic data using Google Maps and TomTom APIs"""
        try:
            traffic_data = {
                'congestion_level': 0,
                'incidents': [],
                'road_closures': [],
                'construction_sites': []
            }
            
            if self.gmaps:
                # Get location coordinates
                geocode_result = self.gmaps.geocode(location)
                if geocode_result:
                    lat = geocode_result[0]['geometry']['location']['lat']
                    lon = geocode_result[0]['geometry']['location']['lng']
                    
                    # Get traffic data from Google Maps
                    traffic_result = self.gmaps.traffic_layer(
                        location=(lat, lon),
                        radius=5000  # 5km radius
                    )
                    
                    if traffic_result:
                        traffic_data['congestion_level'] = self._calculate_congestion_level(traffic_result)
            
            # Get additional traffic data from TomTom
            if self.tomtom_key:
                tomtom_url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
                params = {
                    'key': self.tomtom_key,
                    'point': f"{lat},{lon}",
                    'unit': 'KMPH'
                }
                
                response = requests.get(tomtom_url, params=params)
                if response.status_code == 200:
                    tomtom_data = response.json()
                    incidents = tomtom_data.get('incidents', [])
                    for incident in incidents:
                        traffic_data['incidents'].append({
                            'type': incident.get('type'),
                            'severity': incident.get('severity'),
                            'location': incident.get('location'),
                            'description': incident.get('description')
                        })
            
            return traffic_data
            
        except Exception as e:
            logger.error(f"Error fetching real-time traffic: {str(e)}")
            return None

    def _fetch_construction_data(self, location):
        """Fetch real-time construction and road work data"""
        try:
            construction_data = {
                'active_sites': [],
                'road_works': [],
                'impact_level': 0
            }
            
            if self.gmaps:
                # Get places under construction
                places_result = self.gmaps.places_nearby(
                    location=f"{lat},{lon}",
                    radius=5000,
                    keyword='construction'
                )
                
                if places_result.get('results'):
                    for place in places_result['results']:
                        construction_data['active_sites'].append({
                            'name': place.get('name'),
                            'location': place.get('geometry', {}).get('location'),
                            'type': place.get('types', [])[0] if place.get('types') else 'unknown'
                        })
            
            # Get road work data from TomTom
            if self.tomtom_key:
                tomtom_url = f"https://api.tomtom.com/traffic/services/4/incidentDetails/s3"
                params = {
                    'key': self.tomtom_key,
                    'bbox': f"{lat-0.1},{lon-0.1},{lat+0.1},{lon+0.1}",
                    'fields': '{incidents{type,geometry,properties}}'
                }
                
                response = requests.get(tomtom_url, params=params)
                if response.status_code == 200:
                    roadworks = response.json().get('roadworks', [])
                    for work in roadworks:
                        construction_data['road_works'].append({
                            'type': work.get('type'),
                            'status': work.get('status'),
                            'start_time': work.get('startTime'),
                            'end_time': work.get('endTime'),
                            'impact': work.get('impact')
                        })
            
            # Calculate overall impact level
            construction_data['impact_level'] = self._calculate_construction_impact(
                len(construction_data['active_sites']),
                len(construction_data['road_works'])
            )
            
            return construction_data
            
        except Exception as e:
            logger.error(f"Error fetching construction data: {str(e)}")
            return None

    def _fetch_live_satellite_imagery(self, lat, lon):
        """Fetch and analyze live satellite imagery using Google Earth Engine and Sentinel-2"""
        try:
            satellite_data = {
                'vegetation_index': 0,
                'urban_change': 0,
                'water_bodies': 0,
                'land_use_changes': [],
                'last_update': datetime.now().isoformat()
            }
            
            if self.ee_initialized:
                # Get the most recent Sentinel-2 image
                point = ee.Geometry.Point([lon, lat])
                s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
                    .filterBounds(point) \
                    .filterDate(ee.Date('Now').advance(-30, 'day'), ee.Date('Now')) \
                    .sort('system:time_start', False) \
                    .first()
                
                if s2:
                    # Calculate NDVI
                    nir = s2.select('B8')
                    red = s2.select('B4')
                    ndvi = nir.subtract(red).divide(nir.add(red))
                    
                    # Calculate urban area
                    urban = s2.select('B11')  # SWIR band for urban detection
                    
                    # Get water bodies using NDWI
                    green = s2.select('B3')
                    nir = s2.select('B8')
                    ndwi = green.subtract(nir).divide(green.add(nir))
                    
                    # Get values at point
                    values = ee.Image.cat([ndvi, urban, ndwi]).reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=point,
                        scale=10
                    ).getInfo()
                    
                    satellite_data['vegetation_index'] = values.get('B8', 0)
                    satellite_data['urban_change'] = values.get('B11', 0)
                    satellite_data['water_bodies'] = values.get('ndwi', 0)
            
            # Get high-resolution recent imagery from Sentinel-2
            if self.sentinel_api:
                # Search for the most recent image
                products = self.sentinel_api.query(
                    area=f"POINT({lon} {lat})",
                    date=(datetime.now() - timedelta(days=30), datetime.now()),
                    platformname='Sentinel-2',
                    processinglevel='Level-2A'
                )
                
                if products:
                    # Get the most recent product
                    product_df = self.sentinel_api.to_dataframe(products)
                    latest_product = product_df.iloc[0]
                    
                    # Download and analyze the image
                    title = latest_product['title']
                    cache_file = f"cache/{title}.jp2"
                    
                    if not os.path.exists(cache_file):
                        self.sentinel_api.download(latest_product['uuid'], directory_path='cache')
                    
                    # Analyze land use changes
                    if os.path.exists(cache_file):
                        land_changes = self._analyze_land_use_changes(cache_file)
                        satellite_data['land_use_changes'] = land_changes
            
            return satellite_data
            
        except Exception as e:
            logger.error(f"Error fetching live satellite imagery: {str(e)}")
            return None

    def _analyze_land_use_changes(self, image_path):
        """Analyze land use changes from satellite imagery"""
        try:
            with Image.open(image_path) as img:
                # Convert to numpy array
                img_array = np.array(img)
                
                # Calculate various indices
                changes = []
                
                # Detect urban expansion
                if self._detect_urban_expansion(img_array):
                    changes.append({
                        'type': 'urban_expansion',
                        'severity': 'high',
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Detect deforestation
                if self._detect_deforestation(img_array):
                    changes.append({
                        'type': 'deforestation',
                        'severity': 'medium',
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Detect water body changes
                if self._detect_water_changes(img_array):
                    changes.append({
                        'type': 'water_body_change',
                        'severity': 'medium',
                        'timestamp': datetime.now().isoformat()
                    })
                
                return changes
                
        except Exception as e:
            logger.error(f"Error analyzing land use changes: {str(e)}")
            return []

    def _calculate_congestion_level(self, traffic_data):
        """Calculate traffic congestion level from 0 to 1"""
        try:
            if not traffic_data:
                return 0
                
            # Extract traffic speed ratios
            speed_ratios = []
            for segment in traffic_data:
                if 'speed' in segment and 'freeflow' in segment:
                    ratio = segment['speed'] / segment['freeflow']
                    speed_ratios.append(ratio)
            
            if speed_ratios:
                # Convert to congestion level (1 - avg_ratio)
                return 1 - np.mean(speed_ratios)
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating congestion level: {str(e)}")
            return 0

    def _calculate_construction_impact(self, num_sites, num_roadworks):
        """Calculate construction impact level from 0 to 1"""
        try:
            # Weight factors
            site_weight = 0.6
            roadwork_weight = 0.4
            
            # Normalize counts (assume max 20 sites and 10 roadworks)
            normalized_sites = min(num_sites / 20, 1)
            normalized_roadworks = min(num_roadworks / 10, 1)
            
            # Calculate weighted impact
            impact = (normalized_sites * site_weight) + (normalized_roadworks * roadwork_weight)
            
            return min(impact, 1)  # Cap at 1
            
        except Exception as e:
            logger.error(f"Error calculating construction impact: {str(e)}")
            return 0

    def _detect_urban_expansion(self, img_array):
        """Detect urban expansion from satellite imagery"""
        try:
            # Simple threshold-based detection
            # Assuming urban areas have high reflectance in SWIR band
            swir_threshold = 0.3
            swir_band = img_array[:, :, 11]  # B11 band
            urban_pixels = np.mean(swir_band > swir_threshold)
            
            return urban_pixels > 0.4  # Return True if more than 40% is urban
            
        except Exception as e:
            logger.error(f"Error detecting urban expansion: {str(e)}")
            return False

    def _detect_deforestation(self, img_array):
        """Detect deforestation from satellite imagery"""
        try:
            # Calculate NDVI
            nir_band = img_array[:, :, 7]  # B8 band
            red_band = img_array[:, :, 3]  # B4 band
            
            ndvi = (nir_band - red_band) / (nir_band + red_band)
            
            # Check for significant vegetation loss
            vegetation_pixels = np.mean(ndvi > 0.4)
            
            return vegetation_pixels < 0.3  # Return True if less than 30% is vegetation
            
        except Exception as e:
            logger.error(f"Error detecting deforestation: {str(e)}")
            return False

    def _detect_water_changes(self, img_array):
        """Detect changes in water bodies from satellite imagery"""
        try:
            # Calculate NDWI
            green_band = img_array[:, :, 2]  # B3 band
            nir_band = img_array[:, :, 7]  # B8 band
            
            ndwi = (green_band - nir_band) / (green_band + nir_band)
            
            # Check for significant water presence
            water_pixels = np.mean(ndwi > 0)
            
            return water_pixels > 0.1  # Return True if more than 10% is water
            
        except Exception as e:
            logger.error(f"Error detecting water changes: {str(e)}")
            return False 