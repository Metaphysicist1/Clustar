
import os
import requests

import json
import dotenv
import pandas as pd


dotenv.load_dotenv()

api_endpoint = 'https://api.nasa.gov/neo/rest/v1/feed'
api_key = os.getenv('NASA_API_KEY')

headers = {
    "Accept": "application/json"
}

params = {
    'api_key': api_key,
    'start_date': '2025-04-19',
    'end_date': '2025-04-26'
}

print(f"Requesting URL: {api_endpoint}")
print(f"With Headers: {headers}")
try:
    response = requests.get(api_endpoint, headers=headers, params=params)

    # Check if the request was successful (status code 200 OK)
    response.raise_for_status() # This will raise an HTTPError for bad status codes (like 403, 404, 500)

    print(f"Successfully fetched data! Status Code: {response.status_code}")

    dataframe = pd.DataFrame()
    # --- Parse JSON ---
    data = response.json()
    dfs = []
    # print(json.dumps(data, indent=2)) # Optional: Pretty print the fetched data
    for day in data['near_earth_objects']:
        df_day_data = data['near_earth_objects'][day]
        df_day = pd.json_normalize(df_day_data) # Use json_normalize for potentially nested data
        df_day['observation_date'] = day
        dfs.append(df_day)
        
    dataframe = pd.concat(dfs, axis=0, ignore_index=True)
    print(dataframe[4:7])

    print(dataframe.head())
    print(dataframe.info())
    print(dataframe.shape)
    # data = data['near_earth_objects']['2025-04-19']
    # print(data.keys())
    # print(" data['element_count']", data['element_count'])
    # print("data['near_earth_objects'].keys() ",data['near_earth_objects'].keys())
    # data=data['element_count']
    # df = pd.json_normalize(data) 
    # print(df.head())
    # print(df.info())
    # print(df.shape)
    # df.to_csv('nasa_neo_browse_dataset.csv', index=False, encoding='utf-8')

    # --- Structure Data (Example using pandas) ---
    # The structure depends on the API response. For /neo/browse, it's likely under 'near_earth_objects'
    # if 'near_earth_objects' in data and isinstance(data['near_earth_objects'], list):
    #     neo_list = data['near_earth_objects']
    #     df = pd.json_normalize(neo_list) # Use json_normalize for potentially nested data

    #     print("\nDataFrame created:")
    #     print(df.head())
    #     print(df.info())

    #     # --- Save the Dataset (Example: CSV) ---
    #     csv_filename = "nasa_neo_browse_dataset.csv"
    #     df.to_csv(csv_filename, index=False, encoding='utf-8')
    #     print(f"\nDataset saved to {csv_filename}")

    # else:
    #     print("\nCould not find 'near_earth_objects' list in the response.")
    #     print("Response structure:", json.dumps(data, indent=2))


except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
    print(f"Status Code: {http_err.response.status_code}")
    print(f"Response Text: {http_err.response.text}") # Often contains more details from the API
except requests.exceptions.RequestException as req_err:
    print(f"Error fetching data: {req_err}")
except json.JSONDecodeError:
    print("Error: Could not decode JSON response.")
    print("Response text:", response.text) # Show the raw text if JSON parsing fails
except Exception as e:
    print(f"An unexpected error occurred: {e}")