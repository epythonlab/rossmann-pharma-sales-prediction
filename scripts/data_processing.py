# scripts/data_processing.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataProcessing:
    
    def __init__(self, test_data: pd.DataFrame, train_data: pd.DataFrame):
        """
        Initialize the DataProcessing class with the data.

        Args:
            data (pd.DataFrame): The input DataFrame to process.
        """
        self.train_data = train_data
        self.test_data = test_data
    

    def missing_data_summary(self, data) -> pd.DataFrame:
        """
        Returns a summary of columns with missing data, including count and percentage of missing values.

        Returns:
            pd.DataFrame: A DataFrame with columns 'Missing Count' and 'Percentage (%)' for columns with missing values.
        """
        # Total missing values per column
        missing_data = data.isnull().sum()
        
        # Filter only columns with missing values greater than 0
        missing_data = missing_data[missing_data > 0]
        
        # Calculate the percentage of missing data
        missing_percentage = (missing_data / len(data)) * 100
        
        # Combine the counts and percentages into a DataFrame
        missing_df = pd.DataFrame({
            'Missing Count': missing_data, 
            'Percentage (%)': missing_percentage
        })
        
        # Sort by percentage of missing data
        missing_df = missing_df.sort_values(by='Percentage (%)', ascending=False)
        
        return missing_df

   
    def check_data_types(self):
        """
        Function to check and compare the data types of columns in both training and test datasets.
        
        Args:
        train_data (pd.DataFrame): The training dataset.
        test_data (pd.DataFrame): The test dataset.
        
        Returns:
        None
        """
        # Check data types of the training dataset
        print("Training Dataset Data Types:\n")
        print(self.train_data.dtypes)
        print("\n" + "="*50 + "\n")
        
        # Check data types of the test dataset
        print("Test Dataset Data Types:\n")
        print(self.test_data.dtypes)
        print("\n" + "="*50 + "\n")
        
        # Check for differences in column names and data types
        print("Differences in column names and data types between training and test datasets:\n")
        train_dtypes = self.train_data.dtypes
        test_dtypes = self.test_data.dtypes
        
        for column in train_dtypes.index:
            if column in test_dtypes.index:
                if train_dtypes[column] != test_dtypes[column]:
                    print(f"Data type mismatch for column '{column}':")
                    print(f"Train: {train_dtypes[column]}, Test: {test_dtypes[column]}")
                    print("-" * 50)
            else:
                print(f"Column '{column}' is present in training data but missing in test data.")
        
        for column in test_dtypes.index:
            if column not in train_dtypes.index:
                print(f"Column '{column}' is present in test data but missing in training data.")
    
    def check_outlier(self, variables):
        # Create boxplots for outlier detection
        plt.figure(figsize=(15, 4))
        for i, var in enumerate(variables, 1):
            plt.subplot(1, len(variables), i)
            sns.boxplot(y=self.train_data[var])
            plt.title(f'Boxplot of {var}')
            plt.xlabel('')

        plt.tight_layout()
        plt.show()
        
