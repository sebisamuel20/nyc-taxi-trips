
from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_key:str 
    test_file_key:str 
    artifact_bucket: str



@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    # drift_report_file_path: str



@dataclass
class DataTransformationArtifact:
    transformed_object_file_key:str 
    transformed_train_file_key:str
    transformed_test_file_key:str
    artifact_bucket: str


@dataclass
class RegressionMetricArtifact:
    r2_score:float
    rmse:float
    



@dataclass
class ModelTrainerArtifact:
    trained_model_file_key:str 
    metric_artifact:RegressionMetricArtifact
    artifact_bucket: str



@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    changed_accuracy:float
    s3_model_path:str 
    trained_model_key:str
    artifact_bucket: str



@dataclass
class ModelPusherArtifact:
    bucket_name:str
    s3_model_path:str
