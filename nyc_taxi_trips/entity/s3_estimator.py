
from nyc_taxi_trips.cloud_actions.aws_actions import SimpleStorageService
from nyc_taxi_trips.exception import NycException
from nyc_taxi_trips.entity.estimator import NycModel
from nyc_taxi_trips.utils.main_utils import save_object
import sys,os
from pandas import DataFrame


class NycEstimator:
    """
    This class is used to save and retrieve us_visas model in s3 bucket and to do prediction
    """

    def __init__(self,bucket_name,model_path,):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model:NycModel=None


    def is_model_present(self,model_path):
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except NycException as e:
            print(e)
            return False

    def load_model(self,)->NycModel:
        """
        Load the model from the model_path
        :return:
        """

        return self.s3.load_model(self.model_path,bucket_name=self.bucket_name)

    def save_model(self,from_file, from_bucket, remove:bool=False)->None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        try:
            
            file = self.s3.load_model_from_s3(source_bucket_name=from_bucket,
                                        source_file_key= from_file)
            self.s3.upload_file(file,
                                to_filename=self.model_path,
                                bucket_name=self.bucket_name,
                                remove=remove
                                )
        except Exception as e:
            raise NycException(e, sys)


    def predict(self,dataframe:DataFrame):
        """
        :param dataframe:
        :return:
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise NycException(e, sys)
        
    
    def upload_object_to_folder(self, obj: object, bucket_name, target_key):
        """
        Uploads a file to a specified folder in an S3 bucket.

        Parameters:
        - file_path (str): The path to the file to upload.
        - bucket_name (str): The name of the target S3 bucket.
        - folder_name (str): The folder in the S3 bucket where the file should be uploaded.
        """

        try:
            temp_file_path = 'nyc_taxi_trips\cloud_actions\example.pkl'
            save_object(temp_file_path, obj)
            # Upload the file
            self.s3_client.upload_file(temp_file_path, bucket_name, target_key)
            os.remove(temp_file_path)
        except Exception as e:
            raise NycException(e, sys) from e 
