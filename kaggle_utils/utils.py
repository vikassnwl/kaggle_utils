import pandas as pd
import numpy as np
import math
from scipy.stats import shapiro, anderson
from datetime import datetime
from types import SimpleNamespace
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
    
    def get_cat_col_names(self, ignore_cols_list=[]):
        if not hasattr(self, "cat_cols"):
            self.cat_cols = self.df.select_dtypes(include="object")
        if ignore_cols_list:
            return [col_name for col_name in self.cat_cols.columns if col_name not in ignore_cols_list]
        return list(self.cat_cols.columns)
        
    def get_num_col_names(self, ignore_cols_list=[]):
        if not hasattr(self, "num_cols"):
            self.num_cols = self.df.select_dtypes(include="number")
        if ignore_cols_list:
            return [col_name for col_name in self.num_cols.columns if col_name not in ignore_cols_list]
        return list(self.num_cols.columns)
    
    def value_counts(self, col_name, with_nan=True):
        series = self.df[col_name]
        if with_nan:
            return pd.concat([series.value_counts(), pd.Series([series.isnull().sum()], index=[np.nan])])
        return series.value_counts()
    
    
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
        X_tr = X_tr.astype(X.dtypes)
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


class CustomModelWithPostProcessing(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn estimator that includes a post-processing step.
    
    Parameters:
    -----------
    model: A scikit-learn compatible model that implements fit and predict methods.
    post_processor: A callable that takes the model's predictions as input and returns the post-processed predictions.
    """
    def __init__(self, model, post_processor):
        self.model = model
        self.post_processor = post_processor

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        raw_predictions = self.model.predict(X)
        processed_predictions = self.post_processor(raw_predictions)
        return processed_predictions

    def score(self, X, y):
        return self.model.score(X, y)


def find_optimal_gaussian_transformer(data, method="anderson"):
    valid_methods_list = ["shapiro", "anderson"]

    funcs = [{"func": np.log, "func_name": "log"},
            {"func": np.sqrt, "func_name": "sqrt"},
            {"func": np.cbrt, "func_name": "cbrt"}]
    
    if method == "anderson":
        min_stat = math.inf
        opt_func = None
        opt_func_name = ""
        stats = []
        for func in funcs:
            stat = anderson(func["func"](data)).statistic
            stats.append(stat)
            if stat < min_stat:
                min_stat = stat
                opt_func = func["func"]
                opt_func_name = func["func_name"]

        print(f"Method: {method}")
        print(f"Candidates: {[f['func_name'] for f in funcs]}")
        print(f"Statistics: {stats}")
        print(f"Selected: {opt_func_name} (minimum statistic)")
    
    elif method == "shapiro":
        max_p = -math.inf
        max_func = None
        func_name = ""
        p_values = []
        for func in funcs:
            stat, p = shapiro(func["func"](data))
            p_values.append(p)
            if p > max_p:
                max_p = p
                max_func = func["func"]
                func_name = func["func_name"]

        print(f"Method: {method}")
        print(f"Candidates: {[f['func_name'] for f in funcs]}")
        print(f"P Values: {p_values}")
        print(f"Selected: {func_name} (maximum p value)")
    
    else:
        valid_methods_str = '\n'.join(map(lambda x: '- '+x, valid_methods_list))
        raise Exception(f"{method} is not a valid method. Try one of the following:\n{valid_methods_str}")