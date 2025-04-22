
import pandas as pd

def fetch_sp500_symbols_and_save():
    """
    Fetch the list of S&P 500 symbols from Wikipedia and save to sp500.csv.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    output_file = "/Users/fehmikatar/Desktop/Stock/leader_scan/resources/sp400.csv"

    try:
        # Read the first table on the page
        tables = pd.read_html(url)
        sp500_table = tables[0]  # The first table contains the S&P 500 data

        # Extract the 'Symbol' column
        symbols = sp500_table["Symbol"].tolist()

        # Clean symbols (remove potential whitespace)
        symbols = [s.strip().upper() for s in symbols if isinstance(s, str)]

        # Save to CSV
        pd.DataFrame({"Symbol": symbols}).to_csv(output_file, index=False)
        print(f"Saved {len(symbols)} symbols to {output_file}.")
    except Exception as e:
        print(f"Error fetching or saving S&P 00 symbols: {e}")

if __name__ == "__main__":
    fetch_sp500_symbols_and_save()