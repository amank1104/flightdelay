from flask import Flask, render_template, request, jsonify
import datetime
import re
import requests
import random
import time
import numpy as np
import pandas as pd
import joblib
from dateutil import parser
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# --- API Configuration ---
AERODATABOX_API_KEY = "1692a85913mshcfd2611afa939adp1c1955jsn7c3bbba4e942"
WEATHERAPI_KEY = "6d3628be62544e2aa27231011251109"  # Your WeatherAPI key

# Cache for flight data to avoid excessive API calls
flight_cache = {}
cache_expiry = 300  # 5 minutes

# Load or create ML model
try:
    delay_model = joblib.load('delay_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Loaded pre-trained model")
except:
    print("No pre-trained model found. Creating a new one...")
    delay_model = None
    label_encoders = {}
    scaler = StandardScaler()

def train_delay_model():
    """
    Train or retrain the delay prediction model with new data
    """
    # In a real application, you would collect historical data from APIs or databases
    # For demonstration, we'll create synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic features (removed airline information)
    airports = ['DEL', 'BOM', 'BLR', 'MAA', 'CCU', 'HYD', 'AMD']
    weather_conditions = ['Clear', 'Rain', 'Fog', 'Clouds', 'Storm', 'Thunderstorm']
    
    data = {
        'departure_airport': np.random.choice(airports, n_samples),
        'arrival_airport': np.random.choice(airports, n_samples),
        'scheduled_hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'weather_condition': np.random.choice(weather_conditions, n_samples),
        'temperature': np.random.uniform(0, 40, n_samples),
        'humidity': np.random.uniform(20, 100, n_samples),
        'wind_speed': np.random.uniform(0, 50, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable (delay in minutes) based on some rules
    df['delay'] = 0
    df.loc[df['weather_condition'].isin(['Rain', 'Fog', 'Storm', 'Thunderstorm']), 'delay'] += np.random.randint(15, 120, len(df[df['weather_condition'].isin(['Rain', 'Fog', 'Storm', 'Thunderstorm'])]))
    df.loc[df['scheduled_hour'].between(16, 20), 'delay'] += np.random.randint(5, 45, len(df[df['scheduled_hour'].between(16, 20)]))
    df.loc[df['departure_airport'] == 'DEL', 'delay'] += np.random.randint(0, 30, len(df[df['departure_airport'] == 'DEL']))
    df['delay'] += np.random.randint(0, 20, n_samples)  # Random noise
    
    # Encode categorical variables
    global label_encoders, scaler
    label_encoders = {}
    
    for column in ['departure_airport', 'arrival_airport', 'weather_condition']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    # Scale numerical features
    numerical_features = ['scheduled_hour', 'day_of_week', 'temperature', 'humidity', 'wind_speed']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Split data
    X = df.drop('delay', axis=1)
    y = df['delay']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    global delay_model
    delay_model = RandomForestRegressor(n_estimators=100, random_state=42)
    delay_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = delay_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model trained with MAE: {mae:.2f} minutes")
    
    # Save model
    joblib.dump(delay_model, 'delay_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return delay_model

def get_weather_data(city: str) -> dict:
    """
    Get real-time weather data for a city using WeatherAPI
    """
    # First try to map airport names to cities
    airport_to_city = {
        "Indira Gandhi International Airport": "Delhi",
        "Chhatrapati Shivaji Maharaj International Airport": "Mumbai",
        "Kempegowda International Airport": "Bangalore",
        "Chennai International Airport": "Chennai",
        "Netaji Subhas Chandra Bose International Airport": "Kolkata",
        "Rajiv Gandhi International Airport": "Hyderabad",
        "Sardar Vallabhbhai Patel International Airport": "Ahmedabad",
        "Seoul Incheon": "Seoul", 
        "Beijing Capital": "Beijing",
        "Vancouver": "Vancouver",
        "London Heathrow": "London",
        "New York JFK": "New York",
        "Dubai International": "Dubai",
        "Singapore Changi": "Singapore",
        "Tokyo Narita": "Tokyo"
    }
    
    # Use mapped city if available, otherwise use the original
    actual_city = airport_to_city.get(city, city)
    
    # WeatherAPI URL
    url = "https://api.weatherapi.com/v1/current.json"
    params = {
        "key": WEATHERAPI_KEY,
        "q": actual_city,
        "aqi": "no"
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            weather_condition = data['current']['condition']['text']
            temperature = data['current']['temp_c']
            humidity = data['current']['humidity']
            wind_speed = data['current']['wind_kph']
            return {"status": "success", "data": {
                "condition": weather_condition, 
                "temp": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed
            }}
        else:
            print(f"Weather API error: {response.status_code}, Response: {response.text}")
            return get_mock_weather_data(actual_city)
    except Exception as e:
        print(f"Weather API exception: {e}")
        return get_mock_weather_data(actual_city)

def get_mock_weather_data(city: str) -> dict:
    """
    Fallback mock weather data if API fails
    """
    mock_weather = {
        "Delhi": {"condition": "Fog", "temp": 15, "humidity": 85, "wind_speed": 5},
        "Mumbai": {"condition": "Rain", "temp": 28, "humidity": 90, "wind_speed": 12},
        "Bangalore": {"condition": "Clouds", "temp": 22, "humidity": 75, "wind_speed": 8},
        "Chennai": {"condition": "Clear", "temp": 32, "humidity": 65, "wind_speed": 10},
        "Kolkata": {"condition": "Clear", "temp": 29, "humidity": 70, "wind_speed": 7},
        "Hyderabad": {"condition": "Clear", "temp": 30, "humidity": 60, "wind_speed": 6},
        "Ahmedabad": {"condition": "Clear", "temp": 27, "humidity": 55, "wind_speed": 9},
        "Pune": {"condition": "Clouds", "temp": 26, "humidity": 70, "wind_speed": 8},
        "Goa": {"condition": "Clear", "temp": 31, "humidity": 80, "wind_speed": 11},
        "Jaipur": {"condition": "Clear", "temp": 28, "humidity": 50, "wind_speed": 7},
        "Seoul": {"condition": "Clear", "temp": 18, "humidity": 65, "wind_speed": 10},
        "Beijing": {"condition": "Smoke", "temp": 10, "humidity": 45, "wind_speed": 12},
        "Vancouver": {"condition": "Clouds", "temp": 12, "humidity": 75, "wind_speed": 15},
        "London": {"condition": "Rain", "temp": 8, "humidity": 80, "wind_speed": 20},
        "New York": {"condition": "Clear", "temp": 22, "humidity": 60, "wind_speed": 10},
        "Dubai": {"condition": "Clear", "temp": 35, "humidity": 40, "wind_speed": 15},
        "Singapore": {"condition": "Thunderstorm", "temp": 28, "humidity": 85, "wind_speed": 12},
        "Tokyo": {"condition": "Clear", "temp": 20, "humidity": 70, "wind_speed": 8}
    }
    
    # Try to find a matching city even if not exact
    for known_city in mock_weather:
        if known_city.lower() in city.lower():
            return {"status": "success", "data": mock_weather[known_city]}
    
    return {"status": "success", "data": {"condition": "Clear", "temp": 25, "humidity": 60, "wind_speed": 8}}

def get_flight_data(flight_number: str, date_str: str) -> dict:
    """
    Get real-time flight data from Aerodatabox API
    """
    # Check cache first
    cache_key = f"{flight_number}_{date_str}"
    current_time = time.time()
    if cache_key in flight_cache and current_time - flight_cache[cache_key]['timestamp'] < cache_expiry:
        return flight_cache[cache_key]['data']
    
    # Format flight number for API (remove spaces)
    formatted_flight_number = flight_number.replace(' ', '')
    url = f"https://aerodatabox.p.rapidapi.com/flights/number/{formatted_flight_number}/{date_str}"
    headers = {
        "X-RapidAPI-Key": AERODATABOX_API_KEY,
        "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
    }
    
    try:
        response = requests.get(url, headers=headers)
        print("Aerodatabox API Status:", response.status_code)
        
        if response.status_code == 200:
            data = response.json()
            print("Aerodatabox API Response:", data)  # Debug output
            
            if data and len(data) > 0:
                flight = data[0]
                departure = flight.get("departure", {})
                arrival = flight.get("arrival", {})
                
                # Calculate delays with better error handling
                dep_delay = 0
                arr_delay = 0
                
                # Handle time parsing with better error handling
                scheduled_dep = None
                actual_dep = None
                scheduled_arr = None
                actual_arr = None
                
                # Extract departure times
                if departure.get("scheduledTime") and departure["scheduledTime"].get("local"):
                    try:
                        scheduled_dep = parser.parse(departure["scheduledTime"]["local"])
                    except:
                        scheduled_dep = datetime.datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
                elif departure.get("scheduledTimeLocal"):
                    try:
                        scheduled_dep = parser.parse(departure["scheduledTimeLocal"])
                    except:
                        scheduled_dep = datetime.datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
                
                if departure.get("actualTime") and departure["actualTime"].get("local"):
                    try:
                        actual_dep = parser.parse(departure["actualTime"]["local"])
                    except:
                        actual_dep = scheduled_dep
                elif departure.get("actualTimeLocal"):
                    try:
                        actual_dep = parser.parse(departure["actualTimeLocal"])
                    except:
                        actual_dep = scheduled_dep
                elif departure.get("revisedTime") and departure["revisedTime"].get("local"):
                    try:
                        actual_dep = parser.parse(departure["revisedTime"]["local"])
                    except:
                        actual_dep = scheduled_dep
                
                # Extract arrival times
                if arrival.get("scheduledTime") and arrival["scheduledTime"].get("local"):
                    try:
                        scheduled_arr = parser.parse(arrival["scheduledTime"]["local"])
                    except:
                        if scheduled_dep:
                            scheduled_arr = scheduled_dep + datetime.timedelta(hours=2, minutes=30)
                        else:
                            scheduled_arr = datetime.datetime.now() + datetime.timedelta(hours=2, minutes=30)
                elif arrival.get("scheduledTimeLocal"):
                    try:
                        scheduled_arr = parser.parse(arrival["scheduledTimeLocal"])
                    except:
                        if scheduled_dep:
                            scheduled_arr = scheduled_dep + datetime.timedelta(hours=2, minutes=30)
                        else:
                            scheduled_arr = datetime.datetime.now() + datetime.timedelta(hours=2, minutes=30)
                
                if arrival.get("actualTime") and arrival["actualTime"].get("local"):
                    try:
                        actual_arr = parser.parse(arrival["actualTime"]["local"])
                    except:
                        actual_arr = scheduled_arr
                elif arrival.get("actualTimeLocal"):
                    try:
                        actual_arr = parser.parse(arrival["actualTimeLocal"])
                    except:
                        actual_arr = scheduled_arr
                elif arrival.get("revisedTime") and arrival["revisedTime"].get("local"):
                    try:
                        actual_arr = parser.parse(arrival["revisedTime"]["local"])
                    except:
                        actual_arr = scheduled_arr
                elif arrival.get("runwayTime") and arrival["runwayTime"].get("local"):
                    try:
                        actual_arr = parser.parse(arrival["runwayTime"]["local"])
                    except:
                        actual_arr = scheduled_arr
                
                # Calculate delays if we have both scheduled and actual times
                if scheduled_dep and actual_dep:
                    dep_delay = int((actual_dep - scheduled_dep).total_seconds() / 60)
                
                if scheduled_arr and actual_arr:
                    arr_delay = int((actual_arr - scheduled_arr).total_seconds() / 60)
                
                # Format times for display
                def format_time(dt):
                    return dt.strftime("%Y-%m-%d %H:%M") if dt else "Unknown"
                
                # Prepare flight data
                flight_data = {
                    "flight_number": flight.get("number", "Unknown"),
                    "status": flight.get("status", "Unknown").capitalize(),
                    "departure_airport": departure.get("airport", {}).get("name", "Unknown"),
                    "departure_iata": departure.get("airport", {}).get("iata", "Unknown"),
                    "arrival_airport": arrival.get("airport", {}).get("name", "Unknown"),
                    "arrival_iata": arrival.get("airport", {}).get("iata", "Unknown"),
                    "scheduled_departure": format_time(scheduled_dep),
                    "estimated_departure": format_time(actual_dep) if actual_dep else format_time(scheduled_dep),
                    "scheduled_arrival": format_time(scheduled_arr),
                    "estimated_arrival": format_time(actual_arr) if actual_arr else format_time(scheduled_arr),
                    "departure_delay": dep_delay,
                    "arrival_delay": arr_delay,
                    "terminal": departure.get("terminal", "Unknown"),
                    "gate": departure.get("gate", "Unknown"),
                    "scheduled_departure_datetime": scheduled_dep,
                    "scheduled_arrival_datetime": scheduled_arr
                }
                
                # Cache the result
                flight_cache[cache_key] = {
                    'data': {"status": "success", "data": flight_data},
                    'timestamp': current_time
                }
                
                return {"status": "success", "data": flight_data}
            else:
                print("No flight data found in response")
                return get_mock_flight_data(flight_number, date_str)
        else:
            print(f"Aerodatabox API error: {response.status_code}")
            print(f"Response: {response.text}")
            return get_mock_flight_data(flight_number, date_str)
    except Exception as e:
        print(f"Aerodatabox API exception: {e}")
        return get_mock_flight_data(flight_number, date_str)

def get_mock_flight_data(flight_number: str, date_str: str) -> dict:
    """
    Fallback mock flight data if API fails
    """
    # Use current time for mock data
    now = datetime.datetime.now()
    scheduled_departure = now.replace(hour=8, minute=0, second=0, microsecond=0)
    scheduled_arrival = scheduled_departure + datetime.timedelta(hours=2, minutes=30)
    
    def format_time(dt):
        return dt.strftime("%Y-%m-%d %H:%M")
    
    mock_flights = {
        "CA123": {
            "flight_number": "CA123",
            "status": "Arrived",
            "departure_airport": "Beijing Capital",
            "departure_iata": "PEK",
            "arrival_airport": "Seoul Incheon",
            "arrival_iata": "ICN",
            "scheduled_departure": format_time(scheduled_departure),
            "estimated_departure": format_time(scheduled_departure + datetime.timedelta(minutes=29)),
            "scheduled_arrival": format_time(scheduled_arrival),
            "estimated_arrival": format_time(scheduled_arrival),
            "departure_delay": 29,
            "arrival_delay": 0,
            "terminal": "3",
            "gate": "E20",
            "scheduled_departure_datetime": scheduled_departure,
            "scheduled_arrival_datetime": scheduled_arrival
        },
        "AI101": {
            "flight_number": "AI101",
            "status": "Active",
            "departure_airport": "Indira Gandhi International Airport",
            "departure_iata": "DEL",
            "arrival_airport": "Chhatrapati Shivaji Maharaj International Airport",
            "arrival_iata": "BOM",
            "scheduled_departure": format_time(scheduled_departure),
            "estimated_departure": format_time(scheduled_departure + datetime.timedelta(minutes=15)),
            "scheduled_arrival": format_time(scheduled_arrival),
            "estimated_arrival": format_time(scheduled_arrival + datetime.timedelta(minutes=15)),
            "departure_delay": 15,
            "arrival_delay": 15,
            "terminal": "3",
            "gate": "23",
            "scheduled_departure_datetime": scheduled_departure,
            "scheduled_arrival_datetime": scheduled_arrival
        },
        "6E202": {
            "flight_number": "6E202",
            "status": "Landed",
            "departure_airport": "Kempegowda International Airport",
            "departure_iata": "BLR",
            "arrival_airport": "Indira Gandhi International Airport",
            "arrival_iata": "DEL",
            "scheduled_departure": format_time(scheduled_departure - datetime.timedelta(hours=3)),
            "estimated_departure": format_time(scheduled_departure - datetime.timedelta(hours=3)),
            "scheduled_arrival": format_time(scheduled_arrival - datetime.timedelta(hours=3)),
            "estimated_arrival": format_time(scheduled_arrival - datetime.timedelta(hours=2, minutes=45)),
            "departure_delay": 0,
            "arrival_delay": -15,
            "terminal": "2",
            "gate": "15",
            "scheduled_departure_datetime": scheduled_departure - datetime.timedelta(hours=3),
            "scheduled_arrival_datetime": scheduled_arrival - datetime.timedelta(hours=3)
        },
        "UK810": {
            "flight_number": "UK810",
            "status": "Scheduled",
            "departure_airport": "Chennai International Airport",
            "departure_iata": "MAA",
            "arrival_airport": "Kempegowda International Airport",
            "arrival_iata": "BLR",
            "scheduled_departure": format_time(scheduled_departure + datetime.timedelta(hours=4)),
            "estimated_departure": format_time(scheduled_departure + datetime.timedelta(hours=4)),
            "scheduled_arrival": format_time(scheduled_arrival + datetime.timedelta(hours=4)),
            "estimated_arrival": format_time(scheduled_arrival + datetime.timedelta(hours=4)),
            "departure_delay": 0,
            "arrival_delay": 0,
            "terminal": "1",
            "gate": "8",
            "scheduled_departure_datetime": scheduled_departure + datetime.timedelta(hours=4),
            "scheduled_arrival_datetime": scheduled_arrival + datetime.timedelta(hours=4)
        }
    }
    
    # Try to find a matching flight even if not exact
    for known_flight in mock_flights:
        if known_flight in flight_number:
            return {"status": "success", "data": mock_flights[known_flight]}
    
    # Default mock data
    return {
        "status": "success", 
        "data": {
            "flight_number": flight_number,
            "status": "Scheduled",
            "departure_airport": "Unknown Airport",
            "departure_iata": "XXX",
            "arrival_airport": "Unknown Airport",
            "arrival_iata": "XXX",
            "scheduled_departure": format_time(scheduled_departure),
            "estimated_departure": format_time(scheduled_departure),
            "scheduled_arrival": format_time(scheduled_arrival),
            "estimated_arrival": format_time(scheduled_arrival),
            "departure_delay": 0,
            "arrival_delay": 0,
            "terminal": "Unknown",
            "gate": "Unknown",
            "scheduled_departure_datetime": scheduled_departure,
            "scheduled_arrival_datetime": scheduled_arrival
        }
    }

def predict_future_delay_ml(flight_data, weather_data):
    """
    Predict potential future delays using the ML model
    """
    # If no model exists, train one
    global delay_model
    if delay_model is None:
        delay_model = train_delay_model()
    
    # Prepare features for prediction
    try:
        # Extract features from flight data
        departure_airport = flight_data.get("departure_iata", "Unknown")
        arrival_airport = flight_data.get("arrival_iata", "Unknown")
        
        # Extract time features
        scheduled_departure = flight_data.get("scheduled_departure_datetime")
        if scheduled_departure:
            scheduled_hour = scheduled_departure.hour
            day_of_week = scheduled_departure.weekday()
        else:
            scheduled_hour = 12  # Default to noon
            day_of_week = 0      # Default to Monday
        
        # Extract weather features
        weather_condition = weather_data.get("condition", "Clear")
        temperature = weather_data.get("temp", 25)
        humidity = weather_data.get("humidity", 60)
        wind_speed = weather_data.get("wind_speed", 10)
        
        # Create feature dictionary
        features = {
            'departure_airport': departure_airport,
            'arrival_airport': arrival_airport,
            'scheduled_hour': scheduled_hour,
            'day_of_week': day_of_week,
            'weather_condition': weather_condition,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed
        }
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Encode categorical variables
        for column in ['departure_airport', 'arrival_airport', 'weather_condition']:
            if column in label_encoders:
                try:
                    feature_df[column] = label_encoders[column].transform(feature_df[column])
                except:
                    # If unknown category, use the most common one
                    feature_df[column] = 0
        
        # Scale numerical features
        numerical_features = ['scheduled_hour', 'day_of_week', 'temperature', 'humidity', 'wind_speed']
        feature_df[numerical_features] = scaler.transform(feature_df[numerical_features])
        
        # Make prediction
        predicted_delay = delay_model.predict(feature_df)[0]
        predicted_delay = max(0, predicted_delay)  # Ensure non-negative
        
        # Determine confidence based on feature quality
        confidence = "85%"
        if departure_airport == "Unknown":
            confidence = "65%"
        
        # Generate reason based on features
        reasons = []
        if weather_condition.lower() in ['rain', 'fog', 'storm', 'thunderstorm']:
            reasons.append(f"Poor weather conditions ({weather_condition})")
        
        if scheduled_hour >= 16 and scheduled_hour <= 20:
            reasons.append("Peak travel hours")
        
        if not reasons:
            reasons.append("Normal operating conditions")
        
        reason = " and ".join(reasons) + " may affect arrival time"
        
        return {
            "predicted_delay": int(predicted_delay),
            "confidence": confidence,
            "reason": reason,
            "model_used": "AI Prediction"
        }
    
    except Exception as e:
        print(f"ML prediction error: {e}")
        # Fall back to rule-based prediction if ML fails
        return predict_future_delay_rule_based(flight_data, weather_data)

def predict_future_delay_rule_based(flight_data, weather_data):
    """
    Fallback rule-based prediction if ML model fails
    """
    current_status = flight_data.get("status", "unknown").lower()
    arrival_delay = flight_data.get("arrival_delay", 0)
    weather_condition = weather_data.get("condition", "Clear").lower()
    
    # If flight is already landed/arrived, no future delay
    if current_status in ["landed", "arrived"]:
        return {
            "predicted_delay": arrival_delay,
            "confidence": "100%",
            "reason": "Flight has already landed",
            "model_used": "Rule-based"
        }
    
    # If flight is active, predict based on current conditions
    if current_status == "active":
        # Base prediction on current delay and weather
        if any(cond in weather_condition for cond in ["thunderstorm", "rain", "fog", "snow", "storm"]):
            additional_delay = random.randint(15, 45)
            return {
                "predicted_delay": arrival_delay + additional_delay,
                "confidence": "85%",
                "reason": f"Poor weather conditions ({weather_data.get('condition', 'Unknown')}) may cause additional delays",
                "model_used": "Rule-based"
            }
        else:
            return {
                "predicted_delay": max(0, arrival_delay),  # Ensure non-negative
                "confidence": "75%",
                "reason": "Weather conditions are favorable for on-time arrival",
                "model_used": "Rule-based"
            }
    
    # If flight is scheduled, predict based on weather
    if current_status == "scheduled":
        if any(cond in weather_condition for cond in ["thunderstorm", "rain", "storm"]):
            delay = random.randint(30, 90)
            return {
                "predicted_delay": delay,
                "confidence": "90%",
                "reason": f"Poor weather conditions ({weather_data.get('condition', 'Unknown')}) expected to cause delays",
                "model_used": "Rule-based"
            }
        elif any(cond in weather_condition for cond in ["fog", "snow", "mist"]):
            delay = random.randint(15, 45)
            return {
                "predicted_delay": delay,
                "confidence": "80%",
                "reason": f"{weather_data.get('condition', 'Unknown')} conditions may cause delays",
                "model_used": "Rule-based"
            }
        else:
            return {
                "predicted_delay": 0,
                "confidence": "70%",
                "reason": "Weather conditions are favorable for on-time departure",
                "model_used": "Rule-based"
            }
    
    # Default prediction
    return {
        "predicted_delay": 0,
        "confidence": "50%",
        "reason": "Insufficient data for accurate prediction",
        "model_used": "Rule-based"
    }

def predict_future_delay(flight_data, weather_data):
    """
    Main prediction function that tries ML first, then falls back to rules
    """
    try:
        return predict_future_delay_ml(flight_data, weather_data)
    except Exception as e:
        print(f"ML prediction failed, falling back to rules: {e}")
        return predict_future_delay_rule_based(flight_data, weather_data)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_flight_info', methods=['POST'])
def get_flight_info():
    data = request.get_json()
    flight_number = data.get('flight_number', '').strip().upper()
    date_str = data.get('date', '')
    
    if not flight_number:
        return jsonify({"error": "Flight number is required"}), 400
    
    # If no date provided, use today's date
    if not date_str:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Validate flight number format
    if not re.match(r'^[A-Z0-9]{2,3}-?[0-9]{1,4}[A-Z]?$', flight_number):
        return jsonify({"error": "Invalid flight number format"}), 400
    
    # Ensure proper format (remove any dashes for consistency)
    flight_number = flight_number.replace('-', '')
    
    # Get flight data
    flight_response = get_flight_data(flight_number, date_str)
    if flight_response.get("status") != "success":
        return jsonify({"error": "Failed to fetch flight data"}), 500
    
    flight_data = flight_response.get("data", {})
    
    # Get weather data for departure airport
    departure_city = flight_data.get("departure_airport", "Delhi")
    weather_response = get_weather_data(departure_city)
    if weather_response.get("status") != "success":
        return jsonify({"error": "Failed to fetch weather data"}), 500
    
    weather_data = weather_response.get("data", {})
    
    # Predict future delays
    delay_prediction = predict_future_delay(flight_data, weather_data)
    
    # Prepare response
    result = {
        "flight_data": flight_data,
        "weather_data": weather_data,
        "prediction": delay_prediction
    }
    
    return jsonify(result)

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    """
    Endpoint to manually retrain the ML model
    """
    try:
        train_delay_model()
        return jsonify({"status": "success", "message": "Model retrained successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to retrain model: {str(e)}"}), 500

if __name__ == '__main__':
    # Train model on startup if not already trained
    if delay_model is None:
        train_delay_model()
    
    app.run(debug=True, host='0.0.0.0', port=5001)