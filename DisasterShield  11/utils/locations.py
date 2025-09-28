import json

# Major disaster-prone locations in India with their coordinates
DISASTER_LOCATIONS = {
    "Mumbai": {
        "lat": 19.0760,
        "lon": 72.8777,
        "disaster_types": ["flood", "cyclone", "earthquake"],
        "risk_level": "high"
    },
    "Chennai": {
        "lat": 13.0827,
        "lon": 80.2707,
        "disaster_types": ["cyclone", "flood", "tsunami"],
        "risk_level": "high"
    },
    "Kolkata": {
        "lat": 22.5726,
        "lon": 88.3639,
        "disaster_types": ["cyclone", "flood", "earthquake"],
        "risk_level": "high"
    },
    "Delhi": {
        "lat": 28.6139,
        "lon": 77.2090,
        "disaster_types": ["earthquake", "flood"],
        "risk_level": "medium"
    },
    "Kerala": {
        "lat": 10.8505,
        "lon": 76.2711,
        "disaster_types": ["flood", "landslide"],
        "risk_level": "high"
    },
    "Uttarakhand": {
        "lat": 30.0668,
        "lon": 79.0193,
        "disaster_types": ["landslide", "flood", "earthquake"],
        "risk_level": "high"
    },
    "Gujarat": {
        "lat": 23.2599,
        "lon": 72.5248,
        "disaster_types": ["earthquake", "cyclone", "drought"],
        "risk_level": "high"
    },
    "Assam": {
        "lat": 26.2006,
        "lon": 92.9376,
        "disaster_types": ["flood", "earthquake"],
        "risk_level": "high"
    },
    "Odisha": {
        "lat": 20.9517,
        "lon": 85.0985,
        "disaster_types": ["cyclone", "flood"],
        "risk_level": "high"
    },
    "Andaman and Nicobar Islands": {
        "lat": 11.6234,
        "lon": 92.7265,
        "disaster_types": ["tsunami", "earthquake", "cyclone"],
        "risk_level": "high"
    }
}

def get_location_data(location_name):
    """Get location data by name"""
    return DISASTER_LOCATIONS.get(location_name)

def get_all_locations():
    """Get all disaster-prone locations"""
    return DISASTER_LOCATIONS

def get_locations_by_disaster_type(disaster_type):
    """Get locations affected by a specific disaster type"""
    return {
        name: data for name, data in DISASTER_LOCATIONS.items()
        if disaster_type in data["disaster_types"]
    }

def get_locations_by_risk_level(risk_level):
    """Get locations by risk level"""
    return {
        name: data for name, data in DISASTER_LOCATIONS.items()
        if data["risk_level"] == risk_level
    } 