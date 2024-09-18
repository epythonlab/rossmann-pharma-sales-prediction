import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
class EDAAnalyzer:
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
        train_data['Dataset'] = 'Train'
        test_data['Dataset'] = 'Test'
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
        plt.figure(figsize=(12, 6))

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
