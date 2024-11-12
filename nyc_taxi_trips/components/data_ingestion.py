
import os
import sys
import pandas as pd

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from nyc_taxi_trips.entity.config_entity import DataIngestionConfig
from nyc_taxi_trips.entity.artifact_entity import DataIngestionArtifact
from nyc_taxi_trips.exception import NycException
from nyc_taxi_trips.logger import logging
from nyc_taxi_trips.cloud_actions.aws_actions import SimpleStorageService




class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NycException(e,sys)
        

    
    def export_data_into_feature_store(self)->DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Exporting data from s3 bucket")
            nyc_taxi_data = SimpleStorageService()
            dataframe = nyc_taxi_data.read_csv(filename= "2019-01.csv", bucket_name= self.data_ingestion_config.data_bucket_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path  = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            # logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            # dataframe.to_csv(feature_store_file_path,index=False,header=True)
            # dataframe.to_parquet(feature_store_file_path,index=False)
            
            return dataframe

        except Exception as e:
            raise NycException(e,sys)
        

    def split_data_as_train_test(self,dataframe: DataFrame) ->None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            nyc_taxi_data = SimpleStorageService()
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            # dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            # os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            # train_set.to_parquet(self.data_ingestion_config.training_file_path,index=False)
            # test_set.to_parquet(self.data_ingestion_config.testing_file_path,index=False)
            nyc_taxi_data.write_parquet_to_s3(train_set, self.data_ingestion_config.artifact_bucket_name, self.data_ingestion_config.training_file_key)
            nyc_taxi_data.write_parquet_to_s3(test_set, self.data_ingestion_config.artifact_bucket_name, self.data_ingestion_config.testing_file_key)

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise NycException(e, sys) from e
        


    
    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            dataframe = self.export_data_into_feature_store()

            logging.info("Got the data from s3 bucket")

            self.split_data_as_train_test(dataframe)

            logging.info("Performed train test split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(trained_file_key=self.data_ingestion_config.training_file_key,
            test_file_key=self.data_ingestion_config.testing_file_key)
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise NycException(e, sys) from e
