import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

class Visualyzer:
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        self.train_data = train_data
        self.test_data = test_data

    def check_promotion_distribution(self):
            
        # Compute proportions
        train_promo_dist = self.train_data['Promo'].value_counts(normalize=True)
        test_promo_dist = self.test_data['Promo'].value_counts(normalize=True)
        train_promo2_dist = self.train_data['Promo2'].value_counts(normalize=True)
        test_promo2_dist = self.test_data['Promo2'].value_counts(normalize=True)

        # Combine training and test data into a single DataFrame with a dataset label
        self.train_data['Dataset'] = 'Train'
        self.test_data['Dataset'] = 'Test'
        combined_df = pd.concat([self.train_data, self.test_data])

        # Create contingency tables for 'Promo'
        promo_contingency = pd.crosstab(combined_df['Promo'], combined_df['Dataset'])
        print("Promo Contingency Table:\n", promo_contingency)

        # Perform Chi-square test for 'Promo'
        chi2_promo, p_promo, _, _ = chi2_contingency(promo_contingency)
        print(f"\nChi-square test for Promo: p-value = {p_promo}")

        # Create contingency tables for 'Promo2'
        promo2_contingency = pd.crosstab(combined_df['Promo2'], combined_df['Dataset'])
        print("Promo2 Contingency Table:\n", promo2_contingency)

        # Perform Chi-square test for 'Promo2'
        chi2_promo2, p_promo2, _, _ = chi2_contingency(promo2_contingency)
        print(f"Chi-square test for Promo2: p-value = {p_promo2}")

        # Print proportions
        print("Training Promo Distribution:\n", train_promo_dist)
        print("Testing Promo Distribution:\n", test_promo_dist)
        print("Training Promo2 Distribution:\n", train_promo2_dist)
        print("Testing Promo2 Distribution:\n", test_promo2_dist)

        # Plot distributions
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        sns.barplot(x=train_promo_dist.index, y=train_promo_dist.values, color='blue', alpha=0.6, label='Training')
        sns.barplot(x=test_promo_dist.index, y=test_promo_dist.values, color='red', alpha=0.6, label='Testing')
        plt.title('Promo Distribution')
        plt.xlabel('Promo')
        plt.ylabel('Proportion')
        plt.legend()

        plt.subplot(1, 2, 2)
        sns.barplot(x=train_promo2_dist.index, y=train_promo2_dist.values, color='blue', alpha=0.6, label='Training')
        sns.barplot(x=test_promo2_dist.index, y=test_promo2_dist.values, color='red', alpha=0.6, label='Testing')
        plt.title('Promo2 Distribution')
        plt.xlabel('Promo2')
        plt.ylabel('Proportion')
        plt.legend()

        plt.tight_layout()
        plt.show()
    
    def compare_sales_behavior(self):
        # Filter only the relevant columns and ensure the store is open
        df = self.train_data.reset_index()
        data = df[df['Open'] == 1][['Store', 'Date', 'Sales', 'StateHoliday']]

        # Convert 'Date' to datetime if not already
        data['Date'] = pd.to_datetime(data['Date'])

        # Sort data by store and date
        data = data.sort_values(by=['Store', 'Date'])

        # Identify holiday periods
        data['HolidayPeriod'] = np.where(data['StateHoliday'].isin(['a', 'b', 'c']), 'During Holiday', 'Non-Holiday')

        # Shift rows to capture before and after holidays
        data['BeforeHoliday'] = data['StateHoliday'].shift(-1).isin(['a', 'b', 'c'])
        data['AfterHoliday'] = data['StateHoliday'].shift(1).isin(['a', 'b', 'c'])

        # Define before, during, and after periods
        data['HolidayPeriod'] = np.where(data['BeforeHoliday'], 'Before Holiday', data['HolidayPeriod'])
        data['HolidayPeriod'] = np.where(data['AfterHoliday'], 'After Holiday', data['HolidayPeriod'])

        # Group by HolidayPeriod and calculate average sales
        holiday_sales = data.groupby('HolidayPeriod')['Sales'].mean().reset_index()

        
        # Plot sales behavior before, during, and after holidays
        plt.figure(figsize=(10, 4))
        bars = plt.bar(holiday_sales['HolidayPeriod'], holiday_sales['Sales'], color=['blue', 'orange', 'green'])

        # Add annotations on top of each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')  # 'va' for vertical alignment, 'ha' for horizontal alignment

        # Customize plot
        plt.title('Sales Behavior Before, During, and After Holidays')
        plt.xlabel('Holiday Period')
        plt.ylabel('Average Sales')

        # Show plot
        plt.show()
    
    def seasonal_sales_behavior(self, ascending=True): 
    
        # Filter the dataset for open stores
        df_open = self.train_data[self.train_data['Open'] == 1].copy()  # Use a copy to avoid modifying the original data
        # Convert StateHoliday as str
        df_open['StateHoliday'] = df_open['StateHoliday'].astype(str)
        
        # Group by StateHoliday and calculate the average sales
        seasonal_sales = df_open.groupby('StateHoliday')['Sales'].mean().reset_index()

        # Rename the holidays for better understanding
        seasonal_sales['StateHoliday'] = seasonal_sales['StateHoliday'].replace({
            'a': 'Public Holiday',
            'b': 'Easter Holiday',
            'c': 'Christmas',
            '0': 'No Holiday'
        })

        # Sort by Sales in ascending or descending order based on the argument
        seasonal_sales = seasonal_sales.sort_values(by='Sales', ascending=ascending)

        # Generate color dynamically based on the number of unique categories
        colors = plt.cm.Paired(range(len(seasonal_sales)))

        # Plot the seasonal behavior
        plt.figure(figsize=(8, 4))
        bars = plt.bar(seasonal_sales['StateHoliday'], seasonal_sales['Sales'], color=colors)

        # Annotate the bars with the exact sales numbers
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')

        # Customize the plot
        plt.title('Seasonal Sales Behavior (Christmas, Easter, etc.)')
        plt.xlabel('Holiday Type')
        plt.ylabel('Average Sales')

        plt.show()
    def plot_promo_impact(self):
        """
        Plots the impact of promotions on average sales and customer counts.

        Parameters:
        train_data (DataFrame): The input dataframe containing promotional and sales data.
        """
        # Reset index if needed
        df = self.train_data.reset_index()
        
        # Ensure that 'Promo' and 'Promo2' are of correct type
        df['Promo'] = df['Promo'].astype(bool)
        df['Promo2'] = df['Promo2'].astype(bool)

        # Calculate average sales and customer counts
        promo_sales_avg = df[df['Promo']]['Sales'].mean()
        non_promo_sales_avg = df[~df['Promo']]['Sales'].mean()

        promo2_sales_avg = df[df['Promo2']]['Sales'].mean()
        non_promo2_sales_avg = df[~df['Promo2']]['Sales'].mean()

        promo_customers_avg = df[df['Promo']]['Customers'].mean()
        non_promo_customers_avg = df[~df['Promo']]['Customers'].mean()

        promo2_customers_avg = df[df['Promo2']]['Customers'].mean()
        non_promo2_customers_avg = df[~df['Promo2']]['Customers'].mean()

        # Bar charts
        fig, axs = plt.subplots(2, 2, figsize=(12, 6))

        # Sales for Promo
        axs[0, 0].bar(['Promo', 'Non-Promo'], [promo_sales_avg, non_promo_sales_avg], color=['blue', 'orange'])
        axs[0, 0].set_title('Average Sales: Promo vs Non-Promo')
        axs[0, 0].set_ylabel('Average Sales')

        # Sales for Promo2
        axs[0, 1].bar(['Promo2', 'Non-Promo2'], [promo2_sales_avg, non_promo2_sales_avg], color=['green', 'red'])
        axs[0, 1].set_title('Average Sales: Promo2 vs Non-Promo2')
        axs[0, 1].set_ylabel('Average Sales')

        # Customer Counts for Promo
        axs[1, 0].bar(['Promo', 'Non-Promo'], [promo_customers_avg, non_promo_customers_avg], color=['blue', 'orange'])
        axs[1, 0].set_title('Average Customer Count: Promo vs Non-Promo')
        axs[1, 0].set_ylabel('Average Customer Count')

        # Customer Counts for Promo2
        axs[1, 1].bar(['Promo2', 'Non-Promo2'], [promo2_customers_avg, non_promo2_customers_avg], color=['green', 'red'])
        axs[1, 1].set_title('Average Customer Count: Promo2 vs Non-Promo2')
        axs[1, 1].set_ylabel('Average Customer Count')

        plt.tight_layout()
        plt.show()
    
    def _high_impact_stores(top_n=10):
        # Ensure 'Store', 'Promo', 'Promo2', 'Sales', and 'Customers' are present in the DataFrame
        df = self.train_data.reset_index()
        required_columns = {'Store', 'Promo', 'Promo2', 'Sales', 'Customers'}
        if not required_columns.issubset(df.columns):
            raise KeyError(f"One or more required columns are missing: {required_columns}")
        
        # Filter for stores with Promo active and calculate the mean Sales and Customers by store
        promo_impact = df[df['Promo'] == 1][['Store', 'Sales', 'Customers']].groupby('Store').mean().reset_index()
        
        # Filter for stores with Promo2 active and calculate the mean Sales and Customers by store
        promo2_impact = df[df['Promo2'] == 1][['Store', 'Sales', 'Customers']].groupby('Store').mean().reset_index()
        
        # Merge data for comparison
        common_stores_comparison = pd.merge(promo_impact, promo2_impact, on='Store', suffixes=('_Promo', '_Promo2'))

        # Sort stores by average sales in descending order for both Promo and Promo2
        promo_impact_sorted = promo_impact.sort_values(by='Sales', ascending=False).head(top_n)
        promo2_impact_sorted = promo2_impact.sort_values(by='Sales', ascending=False).head(top_n)
        
        # Define a color palette
        promo_colors = sns.color_palette("Blues", top_n)
        promo2_colors = sns.color_palette("Greens", top_n)
        
        # Plotting the high-impact stores for both Promo and Promo2
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart for Promo with different colors for each bar
        axs[0].bar(promo_impact_sorted['Store'].astype(str), promo_impact_sorted['Sales'], color=promo_colors)
        axs[0].set_title(f'Top {top_n} High Impact Stores: Promo')
        axs[0].set_xlabel('Store')
        axs[0].set_ylabel('Average Sales')
        axs[0].tick_params(axis='x', rotation=45)

        # Bar chart for Promo2 with different colors for each bar
        axs[1].bar(promo2_impact_sorted['Store'].astype(str), promo2_impact_sorted['Sales'], color=promo2_colors)
        axs[1].set_title(f'Top {top_n} High Impact Stores: Promo2')
        axs[1].set_xlabel('Store')
        axs[1].set_ylabel('Average Sales')
        axs[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()