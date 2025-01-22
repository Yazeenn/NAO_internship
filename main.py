import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Data Loading and Cleaning

def load_and_clean_data():
    """
    Loads and cleans the GHG emissions data from the given Excel file.
    """
    file = 'C:/Users/Yasee/Documents/NAO task/provisionalatmoshpericemissionsghg.xlsx'
    GHGdata = pd.ExcelFile(file)

    # Filter relevant sheets
    irrelevant_sheets = ['Table_of_contents', 'Notes']
    relevant_sheets = [sheet for sheet in GHGdata.sheet_names if sheet not in irrelevant_sheets]
    dataframes = {sheet: GHGdata.parse(sheet) for sheet in relevant_sheets}

    # Extract and clean GHG Total tables
    GHGTotal = dataframes['GHG total']
    GHG_table1 = GHGTotal.iloc[4:40, :].reset_index(drop=True)
    GHG_table2 = GHGTotal.iloc[42:78, :].reset_index(drop=True)

    # Clean Table 1
    GHG_table1_cleaned = GHG_table1.iloc[2:, :].reset_index(drop=True)
    GHG_table1_cleaned.columns = GHG_table1.iloc[1, :]
    GHG_table1_cleaned = GHG_table1_cleaned.dropna(how='all', axis=1)
    GHG_table1_cleaned.columns = GHG_table1_cleaned.iloc[0, :]
    GHG_table1_cleaned = GHG_table1_cleaned[1:].reset_index(drop=True)
    GHG_table1_cleaned.rename(columns={GHG_table1_cleaned.columns[0]: "Year"}, inplace=True)

    # Ensure numeric conversion
    GHG_table1_cleaned["Year"] = pd.to_numeric(GHG_table1_cleaned["Year"], errors="coerce")
    GHG_table1_cleaned = GHG_table1_cleaned.dropna(subset=["Year"])
    GHG_table1_cleaned.iloc[:, 1:] = GHG_table1_cleaned.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    GHG_table1_cleaned["Total Emissions"] = GHG_table1_cleaned.iloc[:, 1:].sum(axis=1)

    return GHG_table1_cleaned

# Visualization Functions

def plot_historical_trend(data):
    """Plots the historical trend of total GHG emissions."""
    plt.figure(figsize=(10, 6))
    plt.plot(data["Year"], data["Total Emissions"], marker='o', linestyle='-')
    plt.title("Total Greenhouse Gas Emissions Over Time (1990–2023)", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Total Emissions (Thousand Tonnes of CO2 Equivalent)", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("historical_data.png")
    plt.show()

def plot_historical_with_forecast(data, future_years, predicted_values):
    """Plots the historical trend along with future forecasts."""
    plt.figure(figsize=(10, 6))
    plt.plot(data["Year"], data["Total Emissions"], marker='o', label="Historical Data")
    plt.plot(future_years, predicted_values, 'r--', label="Forecast (2024–2027)")
    plt.title("Total Greenhouse Gas Emissions (1990–2027)", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Total Emissions (Thousand Tonnes of CO2 Equivalent)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("historical_with_forecast.png")
    plt.show()

def plot_industry_trends(data, top_n):
    """Plots emissions trends for the top industries over time."""
    exclude = ["Total Emissions", "Total greenhouse gas emissions"]

    industry_totals = data.iloc[:, 1:].sum(axis=0).sort_values(ascending=False)
    industry_totals = industry_totals[~industry_totals.index.isin(exclude)]
    top_industries = industry_totals.head(top_n).index.tolist()
    selected_data = data[["Year"] + top_industries]

    plt.figure(figsize=(12, 7))
    for industry in top_industries:
        plt.plot(selected_data["Year"], selected_data[industry], marker='o', label=industry)

    plt.title(f"Trends in Greenhouse Gas Emissions for Top {top_n} Industries (1990–2023)", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Emissions (Thousand Tonnes of CO2 Equivalent)", fontsize=12)
    plt.legend(title="Industry", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("industry_trends.png")
    plt.show()

def plot_industry_contributions(data, top_n):
    """Plots the relative contributions of top industries to total emissions."""
    data.iloc[:, 1:] = data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    data.fillna(0, inplace=True)

    data["Total Emissions"] = data.iloc[:, 1:].sum(axis=1)
    percentage_data = data.copy()
    for col in data.columns[1:-1]:
        percentage_data[col] = (data[col] / data["Total Emissions"]) * 100

    industry_totals = data.iloc[:, 1:-1].sum(axis=0).sort_values(ascending=False)
    industry_totals = industry_totals.drop("Total greenhouse gas emissions", errors="ignore")
    top_industries = industry_totals.head(top_n).index.tolist()

    plot_data = percentage_data[["Year"] + top_industries].copy()
    stack_data = [plot_data[industry].values.astype(float) for industry in top_industries]

    plt.figure(figsize=(12, 7))
    plt.stackplot(plot_data["Year"], stack_data, labels=top_industries)
    plt.title(f"Industry Contributions to Total Emissions Over Time (1990–2023)", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Percentage of Total Emissions", fontsize=12)
    plt.legend(title="Industry", fontsize=9, loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("industry_percentages.png")
    plt.show()

# 3. Main Script

if __name__ == "__main__":

    # Load and clean data
    GHG_table1_cleaned = load_and_clean_data()

    # Forecast future emissions
    X = GHG_table1_cleaned["Year"].values.reshape(-1, 1)
    y = GHG_table1_cleaned["Total Emissions"].values
    model = LinearRegression()
    model.fit(X, y)
    future_years = np.array([2024, 2025, 2026, 2027]).reshape(-1, 1)
    predicted_emissions = model.predict(future_years)

    # Generate visualizations
    plot_historical_with_forecast(GHG_table1_cleaned, future_years, predicted_emissions)
    plot_industry_trends(GHG_table1_cleaned,5)
    plot_industry_contributions(GHG_table1_cleaned,5)
