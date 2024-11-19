
from nyc_taxi_trips.entity.config_entity import ModelEvaluationConfig
from nyc_taxi_trips.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import r2_score
from nyc_taxi_trips.exception import NycException
from nyc_taxi_trips.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from nyc_taxi_trips.logger import logging
from nyc_taxi_trips.cloud_actions.aws_actions import SimpleStorageService
import sys
import pandas as pd
from typing import Optional
from nyc_taxi_trips.entity.s3_estimator import NycEstimator
from dataclasses import dataclass
from nyc_taxi_trips.entity.estimator import NycModel
from nyc_taxi_trips.utils.main_utils import drop_columns, read_yaml_file

@dataclass
class EvaluateModelResponse:
    trained_model_r2_score: float
    best_model_r2_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._schema_config = read_yaml_file(file_path= SCHEMA_FILE_PATH)
        except Exception as e:
            raise NycException(e, sys) from e

    def get_best_model(self) -> Optional[NycEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model in production
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            nyc_estimator = NycEstimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if nyc_estimator.is_model_present(model_path=model_path):
                return nyc_estimator
            return None
        except Exception as e:
            raise  NycException(e,sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            drop_cols = self._schema_config['drop_columns']
            eva = SimpleStorageService()
            test_df = eva.read_parquet_from_s3(source_file_key=self.data_ingestion_artifact.test_file_key, source_bucket_name = self.data_ingestion_artifact.artifact_bucket)
            test_df = test_df[test_df['passenger_count'] > 0]
            test_df = test_df[test_df['trip_distance'] > 0]
            test_df = test_df[test_df['fare_amount'] > 0]
            test_df = test_df[test_df['total_amount'] > 0]
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

            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]


            # trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            trained_model_r2_score = self.model_trainer_artifact.metric_artifact.r2_score

            best_model_r2_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_r2_score = r2_score(y, y_hat_best_model)
            
            tmp_best_model_score = 0 if best_model_r2_score is None else best_model_r2_score
            result = EvaluateModelResponse(trained_model_r2_score=trained_model_r2_score,
                                           best_model_r2_score=best_model_r2_score,
                                           is_model_accepted=trained_model_r2_score > tmp_best_model_score,
                                           difference=trained_model_r2_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise NycException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_key=self.model_trainer_artifact.trained_model_file_key,
                changed_accuracy=evaluate_model_response.difference,
                artifact_bucket= self.model_trainer_artifact.artifact_bucket)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise NycException(e, sys) from e
