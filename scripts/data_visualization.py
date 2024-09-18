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
    
    def seasonal_sales_behavior(self):
        
        # Filter the dataset for open stores
        df_open = self.train_data[train_data['Open'] == 1]

        # Group by StateHoliday and calculate the average sales
        seasonal_sales = df_open.groupby('StateHoliday')['Sales'].mean().reset_index()

        # Rename the holidays for better understanding

        seasonal_sales['StateHoliday'] = seasonal_sales['StateHoliday'].astype(str).replace({
            'a': 'Public Holiday',
            'b': 'Easter Holiday',
            'c': 'Christmas',
            '0': 'No Holiday'
        })

        # Plot the seasonal behavior
        plt.figure(figsize=(8, 4))
        bars = plt.bar(seasonal_sales['StateHoliday'], seasonal_sales['Sales'], color=['blue', 'orange', 'green', 'red'])

        # Annotate the bars with the exact sales numbers
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')

        # Customize the plot
        plt.title('Seasonal Sales Behavior (Christmas, Easter, etc.)')
        plt.xlabel('Holiday Type')
        plt.ylabel('Average Sales')

        plt.show()
