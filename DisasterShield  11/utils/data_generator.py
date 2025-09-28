import random
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def generate_resource_data(resource_id=None):
    """Generate sample resource data or get/update specific resource"""
    resource_types = [
        {'name': 'Emergency Vehicles', 'description': 'Ambulances and fire trucks', 'icon': '🚑'},
        {'name': 'Medical Supplies', 'description': 'First aid kits and medical equipment', 'icon': '🏥'},
        {'name': 'Food and Water', 'description': 'Emergency food and water supplies', 'icon': '🍽️'},
        {'name': 'Shelter', 'description': 'Emergency shelters and tents', 'icon': '🏠'},
        {'name': 'Rescue Equipment', 'description': 'Search and rescue equipment', 'icon': '🔧'}
    ]
    
    locations = ['Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Bangalore', 'Hyderabad', 'Ahmedabad', 'Pune', 'Jaipur', 'Surat']
    
    # If resource_id is provided, update or get that specific resource
    if resource_id:
        # In a real application, this would interact with a database
        # For now, we'll generate a new resource with the given ID
        resource_type = random.choice(resource_types)
        location = random.choice(locations)
        status = random.choice(['available', 'in_use'])
        
        return {
            'id': resource_id,
            'type': resource_type['name'],
            'description': resource_type['description'],
            'icon': resource_type['icon'],
            'location': location,
            'status': status,
            'capacity': random.randint(1, 50),
            'last_updated': datetime.now().isoformat(),
            'notes': f'Sample notes for resource {resource_id}'
        }
    
    # Generate a list of sample resources
    resources = []
    for _ in range(20):  # Generate 20 sample resources
        resource_type = random.choice(resource_types)
        location = random.choice(locations)
        status = random.choice(['available', 'in_use'])
        
        resources.append({
            'id': str(random.randint(1000, 9999)),
            'type': resource_type['name'],
            'description': resource_type['description'],
            'icon': resource_type['icon'],
            'location': location,
            'status': status,
            'capacity': random.randint(1, 50),
            'last_updated': datetime.now().isoformat(),
            'notes': f'Sample notes for resource {_}'
        })
    
    logger.info(f"Generated {len(resources)} sample resources")
    return resources

def generate_alert_data():
    """Generate sample alert data"""
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
    
    logger.info(f"Generated {len(alerts)} sample alerts")
    return alerts 