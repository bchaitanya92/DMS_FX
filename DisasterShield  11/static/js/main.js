// Global variables
let currentLocation = null;
let globe = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Initialize modals
    var modalElementList = [].slice.call(document.querySelectorAll('.modal'));
    var modalList = modalElementList.map(function (modalEl) {
        return new bootstrap.Modal(modalEl);
    });

    // Add event listeners for forms
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
    }
});

// Handle prediction form submission
async function handlePredictionSubmit(event) {
    event.preventDefault();
    
    const location = document.getElementById('location').value;
    if (!location) {
        showError('Please select a location');
        return;
    }

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ location })
        });

        const data = await response.json();
        if (data.success) {
            showPredictionResults(data);
        } else {
            showError(data.error || 'Failed to get prediction');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('An error occurred while getting the prediction');
    }
}

// Show prediction results
function showPredictionResults(data) {
    const resultsContainer = document.getElementById('predictionResults');
    if (!resultsContainer) return;

    resultsContainer.innerHTML = `
        <div class="prediction-card">
            <h4>Prediction Results</h4>
            <p><strong>Disaster Type:</strong> ${data.disaster_type}</p>
            <p><strong>Probability:</strong> ${(data.probability * 100).toFixed(2)}%</p>
            
            <div class="weather-info mt-3">
                <h5>Current Weather</h5>
                <p>Temperature: ${data.weather.temperature}°C</p>
                <p>Wind Speed: ${data.weather.wind_speed} m/s</p>
                <p>Rainfall: ${data.weather.rainfall} mm</p>
            </div>

            ${data.trends ? `
                <div class="trends-info mt-3">
                    <h5>Weather Trends</h5>
                    <p>Temperature Trend: ${data.trends.temperature_trend}</p>
                    <p>Rainfall Trend: ${data.trends.rainfall_trend}</p>
                    <p>Wind Speed Trend: ${data.trends.wind_speed_trend}</p>
                </div>
            ` : ''}
        </div>
    `;

    resultsContainer.style.display = 'block';
}

// Show error message
function showError(message) {
    const alertContainer = document.createElement('div');
    alertContainer.className = 'alert alert-danger alert-dismissible fade show';
    alertContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    const container = document.querySelector('.container');
    container.insertBefore(alertContainer, container.firstChild);

    // Auto dismiss after 5 seconds
    setTimeout(() => {
        alertContainer.remove();
    }, 5000);
}

// Update location info panel
function updateLocationInfo(point) {
    const info = document.getElementById('locationInfo');
    if (!info) return;

    document.getElementById('locationName').textContent = point.name;
    document.getElementById('coordinates').textContent = `${point.lat.toFixed(4)}, ${point.lng.toFixed(4)}`;
    document.getElementById('riskLevel').textContent = point.risk_level;
    document.getElementById('riskLevel').className = `risk-${point.risk_level}`;
    document.getElementById('disasterTypes').textContent = point.disaster_types.join(', ');
    info.classList.add('active');
}

// Handle location selection
function handleLocationSelect(location) {
    currentLocation = location;
    updateLocationInfo(location);
}

// Refresh alerts
async function refreshAlerts() {
    try {
        const response = await fetch('/api/alerts');
        const data = await response.json();
        if (data.success) {
            updateAlertsList(data.alerts);
        }
    } catch (error) {
        console.error('Error refreshing alerts:', error);
    }
}

// Update alerts list
function updateAlertsList(alerts) {
    const alertsContainer = document.getElementById('alertsList');
    if (!alertsContainer) return;

    alertsContainer.innerHTML = alerts.map(alert => `
        <div class="alert-card">
            <h5>${alert.type}</h5>
            <p><strong>Location:</strong> ${alert.location}</p>
            <p><strong>Severity:</strong> ${alert.severity}</p>
            <p><strong>Status:</strong> ${alert.status}</p>
            <p><strong>Timestamp:</strong> ${new Date(alert.timestamp).toLocaleString()}</p>
        </div>
    `).join('');
}

// Refresh resources
async function refreshResources() {
    try {
        const response = await fetch('/api/resources');
        const data = await response.json();
        if (data.success) {
            updateResourcesList(data.resources);
        }
    } catch (error) {
        console.error('Error refreshing resources:', error);
    }
}

// Update resources list
function updateResourcesList(resources) {
    const resourcesContainer = document.getElementById('resourcesList');
    if (!resourcesContainer) return;

    resourcesContainer.innerHTML = resources.map(resource => `
        <div class="resource-card">
            <h5>${resource.type}</h5>
            <p><strong>Location:</strong> ${resource.location}</p>
            <p><strong>Status:</strong> ${resource.status}</p>
            <p><strong>Capacity:</strong> ${resource.capacity}</p>
            <p><strong>Last Updated:</strong> ${new Date(resource.last_updated).toLocaleString()}</p>
        </div>
    `).join('');
} 