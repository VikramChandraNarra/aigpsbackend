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

option_prompt1 = """
Analyze the input text and perform the following tasks:
1. Extract the locations mentioned in the input text and list them in a 1D array in the order they appear in the journey. Do not modify the names of the locations. cur_loc if not specified.
2. Determine if the user’s input is ambiguous and requires external options (e.g., for vague terms like 'Chinese restaurant', 'flower shop', etc.).
3. If ambiguity exists, output: "Ambiguous: Needs options."
   If no ambiguity exists, output: "Direct: No options needed."

Examples:

Example 1:
Input: "I'm currently at Union Station Toronto but I need to reach Yorkdale Mall before I go to my class at York University Block A."
Output: 
1D Array: [Union Station Toronto, Yorkdale Mall, York University Block A]
Ambiguity: Direct: No options needed
Explanation: All locations specified.

Example 2:
Input: "I start at home, visit the park, and then go to the library."
Output:
1D Array: [home, park, library]
Ambiguity: Ambiguous: Needs options
Explanation: Which park? Which Library? Unclear exactly where to go.

Example 3:
Input: "I want to go to a Chinese restaurant, then to Union Station."
Output:
1D Array: [cur_loc, Chinese restaurant, Union Station]
Ambiguity: Ambiguous: Needs options
Explanation: Which Chinese Restaurant? Unclear exactly where to go.

Example 4:
Input: "I want to go to Union Station."
Output:
1D Array: [cur_loc, Union Station]
Ambiguity: Direct: No options needed
Explanation: All locations specified.

Now, process this input:
Input: "I am at UTSC and I want to go to the gym, then a Chinese restaurant, and then go to Union Station."
Output:
"""

option_prompt2 = """
Analyze the input text and perform the following tasks:
1. Extract the locations mentioned in the input text and list them in a 1D array in the order they appear in the journey. Do not modify the names of the locations. cur_loc if not specified.
2. Determine if the user’s input is ambiguous and requires external options (e.g., for vague terms like 'Chinese restaurant', 'flower shop', etc.).
3. If ambiguity exists, output: "Ambiguous: Needs options."
   If no ambiguity exists, output: "Direct: No options needed."

Examples:

Example 1:
Input: "I'm currently at Union Station Toronto but I need to reach Yorkdale Mall before I go to my class at York University Block A."
Output: 
1D Array: [Union Station Toronto, Yorkdale Mall, York University Block A]
Ambiguity: Direct: No options needed
Explanation: All locations specified.

Example 2:
Input: "I start at home, visit the park, and then go to the library."
Output:
1D Array: [home, park, library]
Ambiguity: Ambiguous: Needs options
Explanation: Which park? Which Library? Unclear exactly where to go.

Example 3:
Input: "I want to go to a Chinese restaurant, then to Union Station."
Output:
1D Array: [cur_loc, Chinese restaurant, Union Station]
Ambiguity: Ambiguous: Needs options
Explanation: Which Chinese Restaurant? Unclear exactly where to go.

Example 4:
Input: "I want to go to Union Station."
Output:
1D Array: [cur_loc, Union Station]
Ambiguity: Direct: No options needed
Explanation: All locations specified.

Now, process this input:
Input: "I want to go to the skating rink"
Output:
"""

option_prompt3 = """
Analyze the input text and perform the following tasks:
1. Extract the locations mentioned in the input text and list them in a 1D array in the order they appear in the journey. Use 'cur_loc' as the starting location if not explicitly mentioned by the user.
2. Determine if the user’s input is ambiguous and requires external options (e.g., for vague terms like 'Chinese restaurant', 'flower shop', 'skating rink', etc.) by identifying any location that does not specify an exact name or address.
3. If ambiguity exists, output: "Ambiguous: Needs options."
   If no ambiguity exists, output: "Direct: No options needed."
4. Provide a clear explanation for the decision to mark the input as ambiguous or direct.

Examples:

Example 1:
Input: "I'm currently at Union Station Toronto but I need to reach Yorkdale Mall before I go to my class at York University Block A."
Output: 
1D Array: [Union Station Toronto, Yorkdale Mall, York University Block A]
Ambiguity: Direct: No options needed
Explanation: All locations are clearly specified with proper names.

Example 2:
Input: "I start at home, visit the park, and then go to the library."
Output:
1D Array: [home, park, library]
Ambiguity: Ambiguous: Needs options
Explanation: 'park' and 'library' are vague and do not specify exact locations. External options are required.

Example 3:
Input: "I want to go to a Chinese restaurant, then to Union Station."
Output:
1D Array: [cur_loc, Chinese restaurant, Union Station]
Ambiguity: Ambiguous: Needs options
Explanation: 'Chinese restaurant' is vague and does not specify an exact location. External options are required for 'Chinese restaurant.'

Example 4:
Input: "I want to go to Union Station."
Output:
1D Array: [cur_loc, Union Station]
Ambiguity: Direct: No options needed
Explanation: 'Union Station' is a specific location, and no ambiguity exists.

Example 5:
Input: "I want to go to the skating rink."
Output:
1D Array: [cur_loc, skating rink]
Ambiguity: Ambiguous: Needs options
Explanation: 'skating rink' is vague and does not specify an exact location. External options are required for 'skating rink.'

Example 6:
Input: "I am at Starbucks on Main Street and want to go to Central Park."
Output:
1D Array: [Starbucks on Main Street, Central Park]
Ambiguity: Direct: No options needed
Explanation: Both locations are clearly specified.

Now, process this input:
Input: "i want to go to union station"
Output:
"""


# Query the Mistral API
chat_response = client.chat.complete(
    model=model,
    messages=[
        {
            "role": "user",
            "content": option_prompt3,
        },
    ]
)

# Print the response
print("Response JSON:")
print(chat_response.choices[0].message.content)
