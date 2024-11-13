
import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from nyc_taxi_trips.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from nyc_taxi_trips.entity.config_entity import DataTransformationConfig
from nyc_taxi_trips.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from nyc_taxi_trips.exception import NycException
from nyc_taxi_trips.logger import logging
from nyc_taxi_trips.utils.main_utils import read_yaml_file, drop_columns, remove_outliers_iqr
from nyc_taxi_trips.cloud_actions.aws_actions import SimpleStorageService



class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise NycException(e, sys)


    
    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            logging.info("Got numerical cols from schema config")

            numeric_transformer = StandardScaler()


            logging.info("Initialized StandardScaler")

            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']

            logging.info("Initialize PowerTransformer")

            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])
            preprocessor = ColumnTransformer(
                [
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            return preprocessor

        except Exception as e:
            raise NycException(e, sys) from e

    def initiate_data_transformation(self, ) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                nyc_artifact = SimpleStorageService()
                train_df = nyc_artifact.read_parquet_from_s3(source_bucket_name= self.data_ingestion_artifact.artifact_bucket, source_file_key= self.data_ingestion_artifact.trained_file_key)
                test_df = nyc_artifact.read_parquet_from_s3(source_bucket_name= self.data_ingestion_artifact.artifact_bucket, source_file_key= self.data_ingestion_artifact.test_file_key)

                logging.info("Got train features and test features of Training dataset")

                train_df = train_df[train_df['passenger_count'] > 0]
                train_df = train_df[train_df['trip_distance'] > 0]
                train_df = train_df[train_df['fare_amount'] > 0]
                train_df = train_df[train_df['total_amount'] > 0]

                logging.info("Filtered only non zero values into the training set")

                train_df['tpep_pickup_datetime'] = train_df['tpep_pickup_datetime'].astype('datetime64[ns]')
                train_df['tpep_dropoff_datetime'] = train_df['tpep_dropoff_datetime'].astype('datetime64[ns]')
                train_df['duration'] = train_df['tpep_dropoff_datetime'] - train_df['tpep_pickup_datetime']
                train_df['duration'] = train_df['duration'].dt.total_seconds()
                train_df['pickup_hour'] = train_df['tpep_pickup_datetime'].dt.hour
                train_df['pickup_day'] = train_df['tpep_pickup_datetime'].dt.day
                train_df['pickup_day_of_week'] = train_df['tpep_pickup_datetime'].dt.day_of_week 
                train_df['pickup_month'] = train_df['tpep_pickup_datetime'].dt.month
                train_df = train_df[train_df['duration'] > 0]

                drop_cols = self._schema_config['drop_columns']
                num_features = self._schema_config['num_features']

                

                logging.info("drop the columns in drop_cols of Training dataset")

                train_df = drop_columns(df=train_df, cols = drop_cols)

                for column in num_features:
                    train_df = remove_outliers_iqr(train_df, column)


                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]
                


                test_df = test_df[test_df['passenger_count'] > 0]
                test_df = test_df[test_df['trip_distance'] > 0]
                test_df = test_df[test_df['fare_amount'] > 0]
                test_df = test_df[test_df['total_amount'] > 0]

                logging.info("Filtered all non zero values into the Test dataset")

                test_df['tpep_pickup_datetime'] = test_df['tpep_pickup_datetime'].astype('datetime64[ns]')
                test_df['tpep_dropoff_datetime'] = test_df['tpep_dropoff_datetime'].astype('datetime64[ns]')
                test_df['duration'] = test_df['tpep_dropoff_datetime'] - test_df['tpep_pickup_datetime']
                test_df['duration'] = test_df['duration'].dt.total_seconds()
                test_df['pickup_hour'] = test_df['tpep_pickup_datetime'].dt.hour
                test_df['pickup_day'] = test_df['tpep_pickup_datetime'].dt.day
                test_df['pickup_day_of_week'] = test_df['tpep_pickup_datetime'].dt.day_of_week 
                test_df['pickup_month'] = test_df['tpep_pickup_datetime'].dt.month
                test_df = test_df[test_df['duration'] > 0]

                test_df = drop_columns(df=test_df, cols = drop_cols)

                logging.info("drop the columns in drop_cols of Test dataset")

                for column in num_features:
                    test_df = remove_outliers_iqr(test_df, column)


                logging.info("Got train features and test features of Testing dataset")

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                num_features.remove(TARGET_COLUMN)


                logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

                logging.info(
                    "Used the preprocessor object to fit transform the train features"
                )

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                logging.info("Used the preprocessor object to transform the test features")


                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]

                test_arr = np.c_[
                    input_feature_test_arr, np.array(target_feature_test_df)
                ]

                pre = SimpleStorageService()

                pre.upload_object_to_folder(obj=preprocessor, bucket_name=self.data_ingestion_artifact.artifact_bucket, target_key=self.data_transformation_config.transformed_object_file_key)
                pre.upload_array_to_folder(array=train_arr, bucket_name=self.data_ingestion_artifact.artifact_bucket, target_key=self.data_transformation_config.transformed_train_file_key)
                pre.upload_array_to_folder(array=test_arr, bucket_name=self.data_ingestion_artifact.artifact_bucket, target_key=self.data_transformation_config.transformed_test_file_key)

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_key=self.data_transformation_config.transformed_object_file_key,
                    transformed_train_file_key=self.data_transformation_config.transformed_train_file_key,
                    transformed_test_file_key=self.data_transformation_config.transformed_test_file_key
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise NycException(e, sys) from e
