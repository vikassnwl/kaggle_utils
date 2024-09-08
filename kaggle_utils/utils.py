import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer


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
        return list(self.cat_cols.columns)
        
    @property
    def get_num_col_names(self):
        if not hasattr(self, "num_cols"):
            self.num_cols = self.df.select_dtypes(include="number")
        return list(self.num_cols.columns)
    

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


def prepend_alphabets_based_on_mean(df):
    # df = df.copy()
    *cat_cols, target_col = df.columns
    # cat_col, target_col = df.columns
    # Step 1: Apply SimpleImputer to handle missing values
    # Impute the category column with most frequent value
    # cat_imputer = SimpleImputer(strategy='most_frequent')
    # df[[cat_col]] = cat_imputer.fit_transform(df[[cat_col]])
    # df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    # Impute the target column with the median
    # target_imputer = SimpleImputer(strategy='median')
    # df[[target_col]] = target_imputer.fit_transform(df[[target_col]])
    df[target_col] = df[target_col].fillna(df[target_col].median())

    alphabet_mappings = []

    for cat_col in cat_cols:
        # Step 2: Calculate the mean for each category
        category_means = df.groupby(cat_col)[target_col].mean()

        # Step 3: Sort the categories based on the mean in ascending order
        sorted_categories = category_means.sort_values().index

        # Step 4: Assign alphabets (A, B, C, ...) based on the sorted order
        alphabet_mapping = {cat: "Z"*(i//26)+chr(65 + i%26) for i, cat in enumerate(sorted_categories)}
        alphabet_mappings.append({v:k for k, v in alphabet_mapping.items()})

        # Step 5: Prepend the alphabet to each category name in the original DataFrame
        # df[cat_col] = df[cat_col].apply(lambda x: f"{alphabet_mapping[x]}_{x}")
        df[cat_col] = df[cat_col].apply(lambda x: f"{alphabet_mapping[x]}")

    return df[cat_cols], alphabet_mappings


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, unknown_value=-1):
        self.unknown_value = unknown_value
        self.ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=unknown_value)
        self.imputer = SimpleImputer(strategy="most_frequent")

    def fit(self, X, y=None):
        X_tr = self.imputer.fit_transform(X)
        X_tr = pd.DataFrame(X_tr, columns=self.imputer.get_feature_names_out(), index=X.index)
        X_mod, alphabet_mappings = prepend_alphabets_based_on_mean(X_tr)
        self.ord_enc.fit(X_mod)
        for alphabet_mapping, category in zip(alphabet_mappings, self.ord_enc.categories_):
            for i in range(len(category)):
                # self.ord_enc.categories_[0][i] = "_".join(self.ord_enc.categories_[0][i].split("_")[1:])
                category[i] = alphabet_mapping[category[i]]
        return self

    def transform(self, X, y=None):
        X_tr = self.imputer.transform(X)
        X_tr = pd.DataFrame(X_tr, columns=self.imputer.get_feature_names_out(), index=X.index)
        X_transformed = self.ord_enc.transform(X_tr.iloc[:, :-1])
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        return self.ord_enc.get_feature_names_out()
