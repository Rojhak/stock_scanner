import csv
import re
import os
from pathlib import Path

# --- Configuration ---
# It's better to define paths relative to the script or get them from config/args
# Example: Assuming resources are in a sibling 'resources' directory
SCRIPT_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = SCRIPT_DIR / "resources"
# Define input and output filenames
INPUT_FILENAME = "new_aktien.csv"
OUTPUT_FILENAME = "cleaned_aktien.csv"

# Construct full paths
input_file_path = RESOURCES_DIR / INPUT_FILENAME
output_file_path = RESOURCES_DIR / OUTPUT_FILENAME

# --- Regular Expression ---
# Regular expression to match ISINs (International Securities Identification Number)
# Format: 2 letters, 9 alphanumeric chars, 1 digit (check digit)
ISIN_PATTERN = re.compile(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b") # Added capturing group

def extract_names_and_isins(input_path: Path, output_path: Path) -> None:
    """
    Reads an input CSV-like file, extracts company names and associated ISINs,
    and writes them to a cleaned CSV file.

    Assumes the input format has names on lines followed by lines starting with "ISIN"
    containing the ISIN code.

    Args:
        input_path: Path to the input file.
        output_path: Path to the output CSV file.
    """
    extracted_data = []
    current_name = None

    print(f"Processing input file: {input_path}")

    try:
        with open(input_path, "r", encoding="utf-8") as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line: # Skip empty lines
                    continue

                # Check if the line contains an ISIN
                isin_match = ISIN_PATTERN.search(line)

                if isin_match:
                    isin = isin_match.group(1) # Get the captured ISIN
                    if current_name:
                        # Remove potential "ISIN " prefix from the name line if it was captured
                        name_to_store = current_name.replace("ISIN", "").strip()
                        extracted_data.append({"Name": name_to_store, "ISIN": isin})
                        # print(f"  Found: Name='{name_to_store}', ISIN='{isin}'") # Debug
                        current_name = None # Reset name after finding ISIN
                    else:
                        print(f"  Warning: Found ISIN '{isin}' on line {line_num} but no preceding name.")
                # Basic check if line might be a name (not empty, not starting with ISIN pattern text)
                elif not line.startswith("ISIN") and not line.startswith("MARKT-") and not line.startswith("PERF.") and not line.startswith("KGV") and not line.startswith("KAUFEN") and not line.startswith("VERKAUFEN"):
                    # This is likely a name line (or part of it)
                    # Handle potential multi-line names by appending if current_name exists
                    if current_name:
                         current_name += " " + line # Append if name spans multiple lines
                    else:
                         current_name = line
                    # print(f"  Potential Name Line: '{current_name}'") # Debug

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except Exception as e:
        print(f"Error reading input file {input_path}: {e}")
        return

    if not extracted_data:
        print("Warning: No Name/ISIN pairs were extracted.")
        return

    print(f"\nWriting {len(extracted_data)} extracted pairs to: {output_path}")
    try:
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8", newline="") as outfile:
            if extracted_data:
                # Use the keys from the first dictionary item as fieldnames
                fieldnames = extracted_data[0].keys()
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(extracted_data)
        print("Successfully wrote cleaned data.")

    except IOError as e:
        print(f"Error writing to output file {output_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during writing: {e}")

if __name__ == "__main__":
    # Ensure the resources directory exists before running
    if not RESOURCES_DIR.exists():
        print(f"Error: Resources directory not found at {RESOURCES_DIR}")
        print("Please ensure the 'resources' directory exists and contains the input file.")
    else:
        extract_names_and_isins(input_file_path, output_file_path)
