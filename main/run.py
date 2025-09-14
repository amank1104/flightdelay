import requests

API_KEY = "1692a85913mshcfd2611afa939adp1c1955jsn7c3bbba4e942"

# Example: Air India flight AI102 on 12th Sept 2025
url = "https://aerodatabox.p.rapidapi.com/flights/number/AI102/2025-09-12"

headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "aerodatabox.p.rapidapi.com"
}

response = requests.get(url, headers=headers)

print("Status Code:", response.status_code)
if response.status_code == 200:
    data = response.json()
    if data:
        flight = data[0]  # take first result
        dep = flight.get("departure", {})
        arr = flight.get("arrival", {})

        print("✈️ Flight:", flight.get("number"))
        print("From:", dep.get("airport", {}).get("name"), "at", dep.get("scheduledTimeLocal"))
        print("To:", arr.get("airport", {}).get("name"), "at", arr.get("scheduledTimeLocal"))

        # Check for delay
        if dep.get("actualTimeLocal"):
            print("\nDeparture Delay (mins):",
                  (pd.to_datetime(dep["actualTimeLocal"]) - pd.to_datetime(dep["scheduledTimeLocal"])).seconds // 60)
        if arr.get("actualTimeLocal"):
            print("Arrival Delay (mins):",
                  (pd.to_datetime(arr["actualTimeLocal"]) - pd.to_datetime(arr["scheduledTimeLocal"])).seconds // 60)
    else:
        print("❌ No flight data found.")
else:
    print("❌ Error:", response.text)
