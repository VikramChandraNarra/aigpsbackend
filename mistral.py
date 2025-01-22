# testing model
# give me the full address of UTSC
# give me the full address of Union station Toronto

import os
from mistralai import Mistral

# Get the API key from environment variables
api_key = os.environ["MISTRAL_API_KEY"]
model = "ministral-8b-latest"
large_model = "pixtral-large-latest"

# Initialize the Mistral client
client = Mistral(api_key=api_key)

# User prompt for navigation route
user_prompt = """
You are a helpful assistant for providing multimodal routes. I would like you to return the route in JSON format, where each subroute is a JSON object.

{
    route1Info: {
        totalTime: int (total time in minutes),
        distance: "total distance in km",
        description: "One-liner describing the route",
        expression: "Describes the sequential order of the modes of transport taken and the number of transitions (e.g., walking | transit | driving | bicycling)",
        efficiency: "time saved",
        effectiveness: "CO2 emissions approx only for driving; otherwise leave as null",
        health: "total calories burned; otherwise leave as null"
    },
    route1: [
        {
            start: "Starting location full address, as it is on Google Maps",
            end: "Destination location full address, as it is on Google Maps",
            timeTaken: "Duration of this segment (e.g., '10 mins', '25 mins')",
            modeOfTransport: "The mode of transport for this segment (options: 'driving', 'walking', 'bicycling', 'transit')",
            nameOfTransport: "For public transit, provide the name or number of the transport (e.g., 'Bus 22', 'Line 1 Subway'); otherwise leave as an empty string",
            calories: "Estimated calories burned for this segment (if applicable, e.g., for walking or cycling); otherwise leave as null",
            gasUsed: "Estimated gas used in liters (if applicable, e.g., for driving); otherwise leave as null",
            totalCost: "Cost incurred for this segment (e.g., ticket prices, tolls, fuel cost); leave as null if not applicable"
        }
    ]
}

Only fill in start and end, extract only what the user mentions, do not add the rest of the address details, the rest fill with zeros for now.

I want to go to Yorkdale Mall from Varsity Centre.
"""

test1 = "Give me 3 indian restaurants near the UTSC, within 5 kilometres"

test2 = """
Context is in double quotes, actual query is not.
"
Extract the order of origin and destinations into a simple 1d array.
E.g.
I'm currently at a but I need to reach b before I go to my class at c.
Example output:
[a, b, c]

Do not modify the names of a,b and c"

I am at UTSC and I want to go to a flower shop and then go to Union Station.
"""

test3 = """
Extract locations mentioned in the input text and order them in a 1D array as they appear in the journey. 
Do not modify the names of the locations.

Example 1:
Input: "I'm currently at a but I need to reach b before I go to my class at c."
Output: [a, b, c]

Example 2:
Input: "I start at home, visit the park, and then go to the library."
Output: [home, park, library]

Now, process this input:
"I am at UTSC and I want to go to the gym, then a chinese restaurant and then go to Union Station."

Output:

"""

# Query the Mistral API
chat_response = client.chat.complete(
    model=model,
    messages=[
        {
            "role": "user",
            "content": test3,
        },
    ]
)

# Print the response
print("Response JSON:")
print(chat_response.choices[0].message.content)
