# DisasterShield - Disaster Prediction System

DisasterShield is a web-based disaster prediction system that uses machine learning to predict potential disasters based on environmental conditions. The system provides real-time alerts and resource management capabilities.

## Features

- Real-time disaster prediction based on environmental conditions
- Active alerts monitoring and management
- Resource tracking and allocation
- SMS notifications for alerts and confirmations
- Modern, responsive web interface
- Automatic data updates every 30 seconds

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Twilio account for SMS functionality (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/disaster-shield.git
cd disaster-shield
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional, for SMS functionality):
Create a `.env` file in the project root with the following variables:
```
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
disaster-shield/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── data/              # Training data and data generation scripts
├── templates/         # HTML templates
│   └── index.html     # Main web interface
└── utils/             # Utility modules
    ├── data_generator.py  # Sample data generation
    ├── ml_predictor.py    # Machine learning model
    └── sms_handler.py     # SMS notification handling
```

## API Endpoints

- `GET /`: Main web interface
- `POST /api/predict`: Make disaster predictions
- `GET /api/resources`: Get available resources
- `GET /api/alerts`: Get active alerts
- `POST /api/send-alert`: Send alert via SMS
- `POST /api/confirm-alert`: Send confirmation message

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flask web framework
- Bootstrap for the UI components
- scikit-learn for machine learning capabilities
- Twilio for SMS functionality 