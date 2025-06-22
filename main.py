from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import json
from dotenv import load_dotenv
import os
import requests
from typing import Dict, Optional, Tuple

app = Flask(__name__)
CORS(app)  # This will allow requests from any origin

# Load environment variables
load_dotenv()

# Step 2: Function to ask ChatGPT for origin, destination, and waypoints
openai.api_key = os.getenv("OPENAI_API_KEY")

conversation_history = []

def geocode_place(place_name: str, user_lat: Optional[float] = None, user_lon: Optional[float] = None) -> Optional[Dict]:
    """
    Geocode a place name to get its specific address and coordinates.
    Uses Google Geocoding API if available, otherwise falls back to a free alternative.
    """
    try:
        # Try Google Geocoding API first if API key is available
        google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if google_api_key:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                "address": place_name,
                "key": google_api_key
            }
            
            # If user location is provided, bias the search towards that area
            if user_lat and user_lon:
                params["location"] = f"{user_lat},{user_lon}"
                params["radius"] = "50000"  # 50km radius
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data["status"] == "OK" and data["results"]:
                result = data["results"][0]
                return {
                    "formatted_address": result["formatted_address"],
                    "lat": result["geometry"]["location"]["lat"],
                    "lng": result["geometry"]["location"]["lng"]
                }
        
        # Fallback to Nominatim (OpenStreetMap) - free but with rate limits
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": place_name,
            "format": "json",
            "limit": 1
        }
        
        # If user location is provided, add it to the query
        if user_lat and user_lon:
            params["q"] += f" near {user_lat},{user_lon}"
        
        headers = {
            "User-Agent": "AIGPSBackend/1.0"
        }
        
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        if data:
            result = data[0]
            return {
                "formatted_address": result.get("display_name", place_name),
                "lat": float(result["lat"]),
                "lng": float(result["lon"])
            }
        
        return None
    except Exception as e:
        print(f"Geocoding error for {place_name}: {e}")
        return None

def generate_route(input_string: str, user_lat: Optional[float] = None, user_lon: Optional[float] = None):
    global conversation_history
    
    # Create the new API request using the chat model
    conversation_history.append({"role": "user", "content": input_string})

    # Build the system prompt with user location context
    system_prompt = """You are a helpful assistant for providing multimodal routes. I would like you to return the route in JSON format, where each subroute is a JSON object. 

IMPORTANT: When the user's origin location is not specific or clear, assume they are starting from their current location coordinates. Extract place names from their request and return them as specific, recognizable location names (e.g., "Central Park", "Times Square", "Starbucks on 5th Avenue") rather than full addresses.

{
route1Info : {
totalTime: int (total time in minutes),
distance: "total distance in km",
description: "One liner describing route",
expression: "Describes the sequential order of the modes of transport taken and the number if transit (e.g walking | transit | driving | bicycling)",
efficiency: "time saved",
effectiveness: "CO2 emissions approx only for driving; otherwise leave as null",
health: "total calories burned; otherwise leave as null"
},
route1: [
    {
        start: "Starting location name (e.g., 'Central Park', 'Times Square') - NOT full address",
        end: "Destination location name (e.g., 'Empire State Building', 'Brooklyn Bridge') - NOT full address",
        timeTaken: "Duration of this segment (e.g., '10 mins', '25 mins')",
        modeOfTransport: "The mode of transport for this segment (options: 'driving', 'walking', 'bicycling', 'transit')",
        nameOfTransport: "For public transit, provide the name or number of the transport (e.g., 'Bus 22', 'Line 1 Subway'); otherwise leave as an empty string",
        calories: "Estimated calories burned for this segment (if applicable, e.g., for walking or cycling); otherwise leave as null",
        gasUsed: "Estimated gas used in liters (if applicable, e.g., for driving); otherwise leave as null",
        totalCost: "Cost incurred for this segment (e.g., ticket prices, tolls, fuel cost); leave as null if not applicable"
    }
]}

Don't turn the keys into strings, don't include any other text."""

    # Add user location context if available
    if user_lat and user_lon:
        system_prompt += f"\n\nUser's current location: Latitude {user_lat}, Longitude {user_lon}. Use this as the starting point when the origin is not specified."

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": system_prompt
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": input_string
                }
            ]
            }
        ] + conversation_history,
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "json_object"
        }
    )
    
    # Extract and parse the response content
    response_content = response.choices[0].message.content
    try:
        parsed_response = json.loads(response_content)
    except json.JSONDecodeError:
        # If parsing fails, return an error message in JSON format
        parsed_response = {"error": "Invalid response format from OpenAI"}

    # Add assistant's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response_content})

    # Geocode the locations in the response
    if "route1" in parsed_response and isinstance(parsed_response["route1"], list):
        for route_segment in parsed_response["route1"]:
            # Geocode start location
            if "start" in route_segment and route_segment["start"]:
                geocoded_start = geocode_place(route_segment["start"], user_lat, user_lon)
                if geocoded_start:
                    route_segment["start_address"] = geocoded_start["formatted_address"]
                    route_segment["start_lat"] = geocoded_start["lat"]
                    route_segment["start_lng"] = geocoded_start["lng"]
            
            # Geocode end location
            if "end" in route_segment and route_segment["end"]:
                geocoded_end = geocode_place(route_segment["end"], user_lat, user_lon)
                if geocoded_end:
                    route_segment["end_address"] = geocoded_end["formatted_address"]
                    route_segment["end_lat"] = geocoded_end["lat"]
                    route_segment["end_lng"] = geocoded_end["lng"]

    return parsed_response

@app.route('/generate_route', methods=['POST'])
def get_route():
    # Get input data from the request
    input_data = request.json
    user_input = input_data.get('input')
    user_location = input_data.get('user_location', {})

    if not user_input:
        return jsonify({'error': 'Please provide a valid input string'}), 400

    # Extract user location coordinates
    user_lat = user_location.get('lat')
    user_lon = user_location.get('lon')

    # Validate coordinates if provided
    if user_lat is not None and user_lon is not None:
        try:
            user_lat = float(user_lat)
            user_lon = float(user_lon)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid latitude or longitude values'}), 400

    # Generate the route
    route_response = generate_route(user_input, user_lat, user_lon)

    print(route_response)

    # Return the response as JSON
    return jsonify(route_response)

if __name__ == '__main__':
    app.run(debug=True, port=5002)