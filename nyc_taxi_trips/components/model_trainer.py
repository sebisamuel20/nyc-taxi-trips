import sys
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error
from neuro_mf  import ModelFactory

from nyc_taxi_trips.exception import NycException
from nyc_taxi_trips.logger import logging
from nyc_taxi_trips.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object
from nyc_taxi_trips.entity.config_entity import ModelTrainerConfig
from nyc_taxi_trips.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, RegressionMetricArtifact
from nyc_taxi_trips.entity.estimator import NycModel
from nyc_taxi_trips.cloud_actions.aws_actions import SimpleStorageService

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and report of the best model
        
        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Using neuro_mf to get best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            best_model_detail = model_factory.get_best_model(
                X=x_train,y=y_train,base_accuracy=self.model_trainer_config.expected_accuracy
            )
            model_obj = best_model_detail.best_model

            y_pred = model_obj.predict(x_test)
            
            r2 = r2_score(y_test, y_pred) 
            rmse = root_mean_squared_error(y_test, y_pred)  

            metric_artifact = RegressionMetricArtifact(r2_score=r2, rmse=rmse)
            
            return best_model_detail, metric_artifact
        
        except Exception as e:
            raise NycException(e, sys) from e
        

    def initiate_model_trainer(self, ) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            mod = SimpleStorageService()
            train_arr = mod.load_array_from_s3(source_bucket_name= self.data_transformation_artifact.artifact_bucket, source_file_key= self.data_transformation_artifact.transformed_train_file_key)
            test_arr = mod.load_array_from_s3(source_bucket_name= self.data_transformation_artifact.artifact_bucket, source_file_key= self.data_transformation_artifact.transformed_test_file_key)
            
            best_model_detail ,metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            
            preprocessing_obj = mod.load_object_from_s3(source_bucket_name= self.data_transformation_artifact.artifact_bucket, source_file_key=self.data_transformation_artifact.transformed_object_file_key)


            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")

            nyc_model = NycModel(preprocessing_object=preprocessing_obj,
                                       trained_model_object=best_model_detail.best_model)
            logging.info("Created usvisa model object with preprocessor and model")
            logging.info("Created best model file path.")
            mod.upload_object_to_folder(obj=nyc_model, bucket_name= self.data_transformation_artifact.artifact_bucket, target_key= self.model_trainer_config.trained_model_file_key)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_key=self.model_trainer_config.trained_model_file_key,
                artifact_bucket=self.data_transformation_artifact.artifact_bucket,
                metric_artifact=metric_artifact
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise NycException(e, sys) from e