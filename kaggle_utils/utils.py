import pandas as pd
from datetime import datetime


class EDA:
    def __init__(self, df):
        self.df = df

    @property
    def get_null_info(self):
        # Get count of null values in each column
        null_counts = self.df.isnull().sum()
        # Get data types of the columns
        column_types = self.df.dtypes
        # Filter only columns with null values and create a new DataFrame with null counts and types
        null_columns_info = pd.DataFrame(
            {
                "null_count": null_counts[null_counts > 0],
                "dtype": column_types[null_counts > 0],
            }
        )
        return null_columns_info

    @property
    def get_cat_cols(self):
        if not hasattr(self, "cat_cols"):
            self.cat_cols = self.df.select_dtypes(include="object")
        return self.cat_cols
    
    @property
    def get_num_cols(self):
        if not hasattr(self, "num_cols"):
            self.num_cols = self.df.select_dtypes(include="number")
        return self.num_cols
    
    @property
    def get_unique_cat_counts(self):
        if not hasattr(self, "cat_cols"):
            self.cat_cols = self.df.select_dtypes(include="object")
        return self.cat_cols.nunique()
    
    @property
    def get_cat_col_names(self):
        if not hasattr(self, "cat_cols"):
            self.cat_cols = self.df.select_dtypes(include="object")
        return self.cat_cols.columns
        
    @property
    def get_num_col_names(self):
        if not hasattr(self, "num_cols"):
            self.num_cols = self.df.select_dtypes(include="number")
        return self.num_cols.columns
    

def prnt(obj):
    for i, item in enumerate(obj):
        print(f"{i+1}. {item}")
    print()


def get_datetime_str():
    # Get the current datetime
    current_time = datetime.now()
    # Format the datetime string
    datetime_str = current_time.strftime("%Y%m%d_%H%M%S")
    return datetime_str