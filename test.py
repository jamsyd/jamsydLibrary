from fredapi import Fred

# Replace 'YOUR_API_KEY' with your actual FRED API key
fred = Fred(api_key='d755e59a79add62ef412263b9414d1ac')

# Retrieve the unemployment rate series
unemployment_rate = fred.get_series('UNRATE')

print(unemployment_rate)
