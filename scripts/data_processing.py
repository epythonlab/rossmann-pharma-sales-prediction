# scripts/data_processing.py
import pandas as pd

class DataProcessing:
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataProcessing class with the data.

        Args:
            data (pd.DataFrame): The input DataFrame to process.
        """
        self.data = data
    

    def missing_data_summary(self) -> pd.DataFrame:
        """
        Returns a summary of columns with missing data, including count and percentage of missing values.

        Returns:
            pd.DataFrame: A DataFrame with columns 'Missing Count' and 'Percentage (%)' for columns with missing values.
        """
        # Total missing values per column
        missing_data = self.data.isnull().sum()
        
        # Filter only columns with missing values greater than 0
        missing_data = missing_data[missing_data > 0]
        
        # Calculate the percentage of missing data
        missing_percentage = (missing_data / len(self.data)) * 100
        
        # Combine the counts and percentages into a DataFrame
        missing_df = pd.DataFrame({
            'Missing Count': missing_data, 
            'Percentage (%)': missing_percentage
        })
        
        # Sort by percentage of missing data
        missing_df = missing_df.sort_values(by='Percentage (%)', ascending=False)
        
        return missing_df
    
    def handle_missing_data(self, missing_type: str, missing_cols: list) -> pd.DataFrame:
        """
        Handles missing data based on predefined strategies.
        """
        if missing_type == 'high':
            # Drop columns with high missing data
            self.data = self.data.drop(columns=missing_cols, errors='ignore')
        elif missing_type == 'moderate':
            # Impute or drop columns with moderate missing data
            for col in missing_cols:
                if col in self.data.columns:
                    if self.data[col].dtype == 'object':
                        # Impute categorical columns with mode (check if mode exists)
                        if not self.data[col].mode().empty:
                            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
                        else:
                            self.data[col] = self.data[col].fillna('Unknown')  # Default for empty mode
                    else:
                        # Impute numerical columns with median (check if median exists)
                        if not self.data[col].isnull().all():  # Ensure column has some numeric values
                            self.data[col] = self.data[col].fillna(self.data[col].median())
                        else:
                            self.data[col] = self.data[col].fillna(0)  # Default for empty median
        else:
            # Handle low missing data (default)
            for col in missing_cols:
                if col in self.data.columns:
                    if self.data[col].dtype == 'object':
                        if not self.data[col].mode().empty:
                            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
                        else:
                            self.data[col] = self.data[col].fillna('Unknown')  # Default for empty mode
                    else:
                        if not self.data[col].isnull().all():
                            self.data[col] = self.data[col].fillna(self.data[col].median())
                        else:
                            self.data[col] = self.data[col].fillna(0)  # Default for empty median

        return self.data


