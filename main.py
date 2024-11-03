from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import json
from dotenv import load_dotenv
import os



app = Flask(__name__)
CORS(app)  # This will allow requests from any origin


# Step 2: Function to ask ChatGPT for origin, destination, and waypoints
openai.api_key = os.getenv("OPENAI_API_KEY")

conversation_history = []

def generate_route(input_string):

    global conversation_history
    # Create the new API request using the chat model
    conversation_history.append({"role": "user", "content": input_string})

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "You are a helpful assistant for providing multimodal routes. I would like you to return the route in JSON format, where each subroute is a jSON object. \n{\nroute1Info : \n{\ntotalTime: int (total time in minutes),\ndistance: \"total distance in km\"\ndescription: \"One liner describing route\",\nexpression:  \"Describes the sequential order of the modes of transport taken and the number if transit (e.g walking | transit | driving | bicycling),\nefficiency: \"time saved\",\neffectiveness: \"CO2 emissions approx only for driving; otherwise leave as null\",\nhealth: \"total calories burned; otherwise leave as null\"\n},\nroute1: [\n    {\n        start: \"Starting location full address, as it is on google maps\",\n        end: \"Destination location full address, as it is on google maps\",\n        timeTaken: \"Duration of this segment (e.g., '10 mins', '25 mins')\",\n        modeOfTransport: \"The mode of transport for this segment (options: 'driving', 'walking', 'bicycling', 'transit')\",\n        nameOfTransport: \"For public transit, provide the name or number of the transport (e.g., 'Bus 22', 'Line 1 Subway'); otherwise leave as an empty string\",\n        calories: \"Estimated calories burned for this segment (if applicable, e.g., for walking or cycling); otherwise leave as null\",\n        gasUsed: \"Estimated gas used in liters (if applicable, e.g., for driving); otherwise leave as null\",\n        totalCost: \"Cost incurred for this segment (e.g., ticket prices, tolls, fuel cost); leave as null if not applicable\"\n    }\n]}\n\nDon't turn the keys into strings, don't include any other text. \n"
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

    return parsed_response


@app.route('/generate_route', methods=['POST'])
def get_route():
    # Get input data from the request
    input_data = request.json
    user_input = input_data.get('input')

    if not user_input:
        return jsonify({'error': 'Please provide a valid input string'}), 400

    # Generate the route
    route_response = generate_route(user_input)

    print(route_response)

    # Return the response as JSON
    return jsonify(route_response)


if __name__ == '__main__':
    app.run(debug=True)
