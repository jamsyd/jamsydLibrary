import requests
import pandas as pd

# Function to get financial data from Alpha Vantage
def get_financial_data(api_key, symbol, report_type):
    url = f"https://www.alphavantage.co/query?function={report_type}&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    return data

# Function to process the financial statement data
def process_financial_data(data, report_type):
    if report_type == "INCOME_STATEMENT":
        statement = data['annualReports']
    elif report_type == "BALANCE_SHEET":
        statement = data['annualReports']
    elif report_type == "CASH_FLOW":
        statement = data['annualReports']
    else:
        raise ValueError("Invalid report type")
    
    df = pd.DataFrame(statement)
    return df

# Main function to fetch and process data
def main():
    api_key = "76845f9110a7fcc42fbd6a395244693b88403dffc2aa8ddb17443581e00befc4"  # Replace with your Alpha Vantage API key
    symbol = "MARA"  # Example for Marathon Digital Holdings

    # Fetch and process Income Statement
    income_data = get_financial_data(api_key, symbol, "INCOME_STATEMENT")
    income_df = process_financial_data(income_data, "INCOME_STATEMENT")
    print("Income Statement:")
    print(income_df)

    # Fetch and process Balance Sheet
    balance_data = get_financial_data(api_key, symbol, "BALANCE_SHEET")
    balance_df = process_financial_data(balance_data, "BALANCE_SHEET")
    print("Balance Sheet:")
    print(balance_df)

    # Fetch and process Cash Flow Statement
    cash_flow_data = get_financial_data(api_key, symbol, "CASH_FLOW")
    cash_flow_df = process_financial_data(cash_flow_data, "CASH_FLOW")
    print("Cash Flow Statement:")
    print(cash_flow_df)

if __name__ == "__main__":
    main()
