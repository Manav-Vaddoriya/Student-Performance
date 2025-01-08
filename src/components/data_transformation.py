import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationconfig:
    pre_processor_obj_filepath=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
    def get_data_transformer_obj(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoding", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Categorical columns: {categorical_columns}')
            logging.info(f'Numerical columns: {numerical_columns}')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data is done")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_obj()

            target = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Verify that target and input columns exist in data
            if target not in train_df.columns or target not in test_df.columns:
                raise KeyError(f"Target column '{target}' is missing in the dataset")

            input_features_train_df = train_df.drop(columns=[target], axis=1)
            target_features_train_df = train_df[target]

            input_features_test_df = test_df.drop(columns=[target], axis=1)
            target_features_test_df = test_df[target]

            logging.info(
                "Applying preprocessing object on training and testing dataframes."
            )

            ip_feature_train_array = preprocessing_obj.fit_transform(input_features_train_df)
            ip_feature_test_array = preprocessing_obj.transform(input_features_test_df)

            train_array = np.c_[ip_feature_train_array, np.array(target_features_train_df)]
            test_array = np.c_[ip_feature_test_array, np.array(target_features_test_df)]

            save_object(
                file_path=self.data_transformation_config.pre_processor_obj_filepath,
                obj=preprocessing_obj
            )

            return (
                train_array,
                test_array,
                self.data_transformation_config.pre_processor_obj_filepath,
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_transformation = DataTransformation()
    preprocessor = data_transformation.get_data_transformer_obj()
    save_object("artifacts/preprocessor.pkl", preprocessor)

