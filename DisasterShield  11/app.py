import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
from utils.data_generator import generate_resource_data, generate_alert_data
from utils.ml_predictor import DisasterPredictor
from utils.sms_handler import SMSHandler
import pandas as pd
import uuid
from utils.weather_fetcher import WeatherDataFetcher
from utils.time_series_handler import TimeSeriesHandler
import random

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
for directory in ['data', 'models', 'templates/errors']:
    os.makedirs(directory, exist_ok=True)

# Initialize Flask app
app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = os.urandom(24)

# Initialize components
logger.info("Initializing DisasterPredictor...")
predictor = DisasterPredictor()
logger.info("DisasterPredictor initialized successfully")

logger.info("Initializing SMSHandler...")
sms_handler = SMSHandler()
logger.info("SMSHandler initialized successfully")

logger.info("Initializing WeatherDataFetcher...")
weather_fetcher = WeatherDataFetcher()
logger.info("WeatherDataFetcher initialized successfully")

logger.info("Initializing TimeSeriesHandler...")
time_series_handler = TimeSeriesHandler()
logger.info("TimeSeriesHandler initialized successfully")

class SessionData:
    def __init__(self):
        self.alerts = []
        self.resources = generate_resource_data()
        self.evacuation_data = {}

session_data = SessionData()

# Generate sample alerts and resources
def generate_sample_alerts():
    locations = ['Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Bangalore', 'Hyderabad', 'Ahmedabad', 'Pune', 'Jaipur', 'Surat']
    disaster_types = ['flood', 'storm', 'earthquake', 'fire']
    alerts = []
    
    for _ in range(5):  # Generate 5 sample alerts
        location = random.choice(locations)
        disaster_type = random.choice(disaster_types)
        severity = random.choice(['high', 'medium', 'low'])
        
        # Generate random confirmation stats
        total_sent = random.randint(100, 500)
        confirmed = random.randint(50, total_sent)
        pending = total_sent - confirmed
        
        alerts.append({
            'id': str(random.randint(1000, 9999)),
            'location': location,
            'disaster_type': disaster_type,
            'severity': severity,
            'message': f"Alert: {disaster_type.title()} warning for {location}. Severity: {severity}",
            'timestamp': (datetime.now() - timedelta(hours=random.randint(0, 24))).isoformat(),
            'confirmation_stats': {
                'total_sent': total_sent,
                'confirmed': confirmed,
                'pending': pending
            }
        })
    
    return alerts

def generate_sample_resources():
    resource_types = [
        {'name': 'Emergency Vehicles', 'description': 'Ambulances and fire trucks'},
        {'name': 'Medical Supplies', 'description': 'First aid kits and medical equipment'},
        {'name': 'Food and Water', 'description': 'Emergency food and water supplies'},
        {'name': 'Shelter', 'description': 'Emergency shelters and tents'},
        {'name': 'Rescue Equipment', 'description': 'Search and rescue equipment'}
    ]
    
    locations = ['Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Bangalore', 'Hyderabad', 'Ahmedabad', 'Pune', 'Jaipur', 'Surat']
    resources = []
    
    for resource in resource_types:
        location = random.choice(locations)
        status = random.choice(['available', 'limited', 'unavailable'])
        
        resources.append({
            'id': str(random.randint(1000, 9999)),
            'name': resource['name'],
            'description': resource['description'],
            'location': location,
            'status': status,
            'quantity': random.randint(1, 50) if status != 'unavailable' else 0
        })
    
    return resources

@app.route('/')
def index():
    """Home page with weather dashboard"""
    try:
        # Get all locations
        locations = weather_fetcher.get_all_locations()
        
        # Format locations data
        formatted_locations = {}
        for location in locations:
            try:
                # Get weather data for each location
                weather_data = weather_fetcher.get_weather(location)
                
                # Get risk level and disaster types
                prediction = predictor.predict_with_weather(location)
                # Handle both dictionary and object responses from predictor
                if isinstance(prediction, dict):
                    probability = prediction.get('probability', 0)
                    disaster_types = prediction.get('disaster_types', [])
                else:
                    probability = getattr(prediction, 'probability', 0)
                    disaster_types = getattr(prediction, 'disaster_types', [])
                
                risk_level = "high" if probability > 0.7 else "medium" if probability > 0.4 else "low"
                
                # Format the data
                formatted_locations[location] = {
                    'lat': weather_data['lat'],
                    'lon': weather_data['lon'],
                    'weather': {
                        'temperature': weather_data['temperature'],
                        'wind_speed': weather_data['wind_speed'],
                        'rainfall': weather_data['rainfall']
                    },
                    'risk_level': risk_level,
                    'disaster_types': disaster_types
                }
            except Exception as e:
                logger.error(f"Error processing location {location}: {str(e)}")
                continue
        
        return render_template('index.html', locations=formatted_locations)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return render_template('error.html', error=str(e))

@app.route('/predictions')
def predictions():
    """Predictions page with map and location selection"""
    try:
        # Get all locations
        locations = weather_fetcher.get_all_locations()
        
        # Format locations data
        formatted_locations = {}
        for location in locations:
            # Get weather data for each location
            weather_data = weather_fetcher.get_weather(location)
            
            # Format the data
            formatted_locations[location] = {
                'lat': weather_data['lat'],
                'lon': weather_data['lon']
            }
        
        return render_template('predictions.html', locations=formatted_locations)
    except Exception as e:
        logger.error(f"Error in predictions route: {str(e)}")
        return render_template('error.html', error=str(e))

@app.route('/alerts')
def alerts():
    """Render the alerts page with both alerts and resources"""
    # Get current weather data for all locations
    locations = weather_fetcher.get_all_locations()
    alerts = []
    resources = []
    
    # Generate alerts based on current weather conditions
    for location in locations:
        # Get current weather data
        weather_data = weather_fetcher.get_weather(location)
        
        # Generate alerts based on weather conditions
        if weather_data['rainfall'] > 50:  # Heavy rainfall
            alerts.append({
                'id': str(uuid.uuid4()),
                'location': location,
                'disaster_type': 'flood',
                'severity': 'high' if weather_data['rainfall'] > 100 else 'medium',
                'message': f"Flood Alert: Heavy rainfall ({weather_data['rainfall']}mm) in {location}",
                'timestamp': datetime.now().isoformat(),
                'weather_data': weather_data
            })
        
        if weather_data['wind_speed'] > 30:  # Strong winds
            alerts.append({
                'id': str(uuid.uuid4()),
                'location': location,
                'disaster_type': 'storm',
                'severity': 'high' if weather_data['wind_speed'] > 40 else 'medium',
                'message': f"Storm Alert: Strong winds ({weather_data['wind_speed']} km/h) in {location}",
                'timestamp': datetime.now().isoformat(),
                'weather_data': weather_data
            })
    
    # Generate resources based on active alerts
    resource_types = [
        {'name': 'Emergency Vehicles', 'description': 'Ambulances and fire trucks', 'icon': '🚑'},
        {'name': 'Medical Supplies', 'description': 'First aid kits and medical equipment', 'icon': '🏥'},
        {'name': 'Food and Water', 'description': 'Emergency food and water supplies', 'icon': '🍽️'},
        {'name': 'Shelter', 'description': 'Emergency shelters and tents', 'icon': '🏠'},
        {'name': 'Rescue Equipment', 'description': 'Search and rescue equipment', 'icon': '🔧'}
    ]
    
    for resource in resource_types:
        # Assign resources to locations with active alerts
        location = random.choice([alert['location'] for alert in alerts]) if alerts else random.choice(locations)
        status = 'available' if random.random() > 0.3 else 'limited'
        
        resources.append({
            'id': str(uuid.uuid4()),
            'name': resource['name'],
            'description': resource['description'],
            'icon': resource['icon'],
            'location': location,
            'status': status,
            'quantity': random.randint(1, 50) if status != 'unavailable' else 0,
            'last_updated': datetime.now().isoformat()
        })
    
    return render_template('alerts.html', alerts=alerts, resources=resources)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        location = data.get('location')
        
        if not location:
            return jsonify({'success': False, 'error': 'Location not provided'})
        
        # Get current weather data
        weather_data = weather_fetcher.get_weather(location)
        
        # Calculate risk factors based on weather conditions
        risk_factors = {
            'Temperature Risk': 'High' if weather_data['temperature'] > 35 else 'Medium' if weather_data['temperature'] > 30 else 'Low',
            'Wind Speed Risk': 'High' if weather_data['wind_speed'] > 30 else 'Medium' if weather_data['wind_speed'] > 20 else 'Low',
            'Rainfall Risk': 'High' if weather_data['rainfall'] > 50 else 'Medium' if weather_data['rainfall'] > 25 else 'Low'
        }
        
        # Determine disaster type and probability based on risk factors
        high_risk_count = sum(1 for risk in risk_factors.values() if risk == 'High')
        medium_risk_count = sum(1 for risk in risk_factors.values() if risk == 'Medium')
        
        if high_risk_count >= 2:
            if weather_data['rainfall'] > 50:
                disaster_type = 'flood'
                probability = 0.8
            elif weather_data['wind_speed'] > 30:
                disaster_type = 'storm'
                probability = 0.75
            else:
                disaster_type = 'heat_wave'
                probability = 0.7
        elif medium_risk_count >= 2:
            if weather_data['rainfall'] > 25:
                disaster_type = 'flood'
                probability = 0.5
            elif weather_data['wind_speed'] > 20:
                disaster_type = 'storm'
                probability = 0.45
            else:
                disaster_type = 'heat_wave'
                probability = 0.4
        else:
            disaster_type = 'no_disaster'
            probability = 0.1
        
        return jsonify({
            'success': True,
            'disaster_type': disaster_type,
            'probability': probability,
            'weather_data': {
                'temperature': weather_data['temperature'],
                'wind_speed': weather_data['wind_speed'],
                'rainfall': weather_data['rainfall']
            },
            'risk_factors': risk_factors
        })
            
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/request-resource', methods=['POST'])
def request_resource():
    """Handle resource requests"""
    try:
        data = request.get_json()
        resource_id = data.get('resource_id')
        
        if not resource_id:
            return jsonify({'success': False, 'error': 'Resource ID not provided'})
        
        # In a real implementation, this would handle the resource request
        return jsonify({'success': True, 'message': 'Resource request submitted successfully'})
        
    except Exception as e:
        logger.error(f"Error in resource request endpoint: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/confirm-alert', methods=['POST'])
def confirm_alert():
    """Handle alert confirmations"""
    try:
        data = request.get_json()
        alert_id = data.get('alert_id')
        
        if not alert_id:
            return jsonify({'success': False, 'error': 'Alert ID not provided'})
        
        # In a real implementation, this would update the alert confirmation status
        return jsonify({'success': True, 'message': 'Alert confirmation recorded'})
        
    except Exception as e:
        logger.error(f"Error in alert confirmation endpoint: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/globe')
def globe():
    """Interactive globe view"""
    try:
        locations = weather_fetcher.get_all_locations()
        globe_data = []
        
        for location in locations:
            weather_data = weather_fetcher.get_weather(location)
            prediction = predictor.predict_with_weather(location)
            
            # Calculate risk level based on prediction
            risk_level = 0.0
            if prediction['disaster_types']:
                # If there are potential disasters, set risk level based on the number of types
                risk_level = min(1.0, len(prediction['disaster_types']) * 0.3)
            
            globe_data.append({
                'name': location,
                'lat': weather_data['lat'],
                'lon': weather_data['lon'],
                'temp': weather_data['temperature'],
                'wind': weather_data['wind_speed'],
                'rainfall': weather_data['rainfall'],
                'risk_level': risk_level,
                'disaster_types': prediction['disaster_types']
            })
        
        return render_template('globe.html', locations=globe_data)
    except Exception as e:
        logger.error(f"Error in globe view: {str(e)}")
        return render_template('error.html', error=str(e))

@app.route('/resources')
def resources():
    """Resource management page"""
    try:
        # Get all resources
        resources = generate_resource_data()
        return render_template('resources.html', resources=resources)
    except Exception as e:
        logger.error(f"Error in resources route: {str(e)}")
        return render_template('error.html', error=str(e))

@app.route('/api/update_resource_status', methods=['POST'])
def update_resource_status():
    """Update resource status API endpoint"""
    try:
        data = request.get_json()
        resource_id = data.get('resource_id')
        
        if not resource_id:
            return jsonify({'success': False, 'error': 'Resource ID is required'})
        
        # Update resource status
        success = generate_resource_data(resource_id)
        
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Failed to update resource status'})
    except Exception as e:
        logger.error(f"Error updating resource status: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/resource_details', methods=['POST'])
def resource_details():
    """Get resource details API endpoint"""
    try:
        data = request.get_json()
        resource_id = data.get('resource_id')
        
        if not resource_id:
            return jsonify({'success': False, 'error': 'Resource ID is required'})
        
        # Get resource details
        resource = generate_resource_data(resource_id)
        
        if resource:
            return jsonify({'success': True, 'resource': resource})
        else:
            return jsonify({'success': False, 'error': 'Resource not found'})
    except Exception as e:
        logger.error(f"Error getting resource details: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('errors/500.html'), 500

if __name__ == '__main__':
    try:
        logger.info("Starting Flask application...")
        app.run(host='127.0.0.1', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start Flask application: {str(e)}", exc_info=True)
        raise 