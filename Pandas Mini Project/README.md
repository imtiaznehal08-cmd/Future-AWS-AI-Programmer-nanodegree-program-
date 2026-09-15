# Pandas-Mini-Project: Stock Data Statistics

## Project Overview
*This project was completed as part of the **AWS Future AI Programmer nanodegree program**.* 

This repository contains a Jupyter Notebook designed to load, analyze, and visualize historical stock data. The analysis focuses on three major tech companies: Google, Apple, and Amazon. Using Python and Pandas, the project demonstrates how to perform data manipulation, calculate basic statistics, and compute moving averages.

## Files Included
The following data files were downloaded from Yahoo Finance and are required in the workspace to run the analysis:
* **`GOOG.csv`**: Contains Google stock data.
* **`AAPL.csv`**: Contains Apple stock data.
* **`AMZN.csv`**: Contains Amazon stock data.
* **`Statistics from Stock Data - Solution.ipynb`**: The primary Jupyter Notebook containing the executable Python code.

Each CSV file originally contains 7 columns: Date, Open, High, Low, Close, Adj_Close, and Volume.

## Features and Analysis
The notebook walks through several key data science operations:
* **Data Loading**: Reads CSV files into a Pandas DataFrame, extracting specifically the `Date` and `Adj Close` columns.
* **Data Cleaning**: Automatically parses dates as the DataFrame index, renames columns to match the respective stock names, and drops any rows containing `NaN` (missing) values.
* **Statistical Calculations**: Computes the average (mean), median, standard deviation, and correlation across the individual stock prices.
* **Rolling Statistics**: Calculates a 150-day rolling mean (moving average) for Google's stock.
* **Visualization**: Utilizes Matplotlib to plot the raw stock price alongside its moving average for clear visual comparison.

## Author

**Nehal Imtiaz**  
Udacity Future AWS AI Programmer Nanodegree Scholar
