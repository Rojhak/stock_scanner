# Stock Copy Project (Leader Scan)

## Overview

This project implements a stock screening tool based on technical and potentially fundamental criteria, inspired by methodologies like Minervini's. It scans stock universes (e.g., S&P 500, S&P 400) to identify potential trading setups like Stage 2 entries, MA crossovers, and general market leadership based on relative strength.

## Features

* Fetches historical price data using `yfinance`.
* Calculates various technical indicators (MAs, ATR, RS, RSI, etc.).
* Scores stocks based on multiple technical and (optionally) fundamental criteria.
* Identifies specific setup types (Stage 2, MA Cross, Leader).
* Calculates basic risk/reward metrics for potential trades.
* Supports different stock universes via CSV files.
* Optional alerting via Slack and Email.
* Trade logging functionality.
* Caching mechanism for price data to speed up subsequent runs.

## Structure

* `leader_scan/`: Main package directory.
    * `main.py`: Entry point and orchestration logic.
    * `data.py`: Data fetching (prices, fundamentals, universe loading) and caching.
    * `indicators.py`: Technical indicator calculations.
    * `filters.py`: Boolean filter functions for setup detection.
    * `scorer.py`: Scoring logic and setup classification.
    * `screener.py`: Stage 2 specific screening logic.
    * `advanced_screener.py`: Script for running combined scans.
    * `trade_logger.py`: Trade journaling functions.
    * `alert.py`: Slack and Email alert functions.
    * `config.py`: Configuration management.
    * `resources/`: Contains CSV files for stock universes (e.g., `sp500.csv`).
    * `.cache/`: Directory for storing cached price data (auto-created).
* `tests/`: Unit tests.
* `requirements.txt`: Project dependencies.
* `README.md`: This file.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Stock copy
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configuration (Optional):**
    * Create a JSON file at `~/.leader_scan.json` to override default settings in `config.py`.
    * Alternatively, set environment variables prefixed with `LS_` (e.g., `export LS_SLACK_WEBHOOK="your_webhook_url"`). See `config.py` for details.

## Usage

### Basic Scan (from `main.py`)

Run the default leadership scan (SP1500 universe, top 20 leaders):
```bash
python -m leader_scan.main 
