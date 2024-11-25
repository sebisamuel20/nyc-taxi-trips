import os
import sys

import numpy as np
import pandas as pd
from nyc_taxi_trips.entity.config_entity import NycTaxiTripPredictorConfig
from nyc_taxi_trips.entity.s3_estimator import NycEstimator
from nyc_taxi_trips.exception import NycException
from nyc_taxi_trips.logger import logging
from nyc_taxi_trips.utils.main_utils import read_yaml_file
from pandas import DataFrame


class NycData:
    def __init__(self,
                vendorid,
                passenger_count,
                trip_distance,
                ratecodeid,
                pulocationid,
                dolocationid,
                payment_type,
                extra,
                mta_tax,
                tip_amount,
                tolls_amount,
                improvement_surcharge,
                duration,
                pickup_hour,
                pickup_day,
                pickup_day_of_week,
                pickup_month

                ):
        """
        NycTrip Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.vendorid = vendorid
            self.pickup_hour = pickup_hour
            self.pickup_day = pickup_day
            self.passenger_count = passenger_count
            self.trip_distance = trip_distance
            self.ratecodeid = ratecodeid
            self.pulocationid = pulocationid
            self.dolocationid = dolocationid
            self.payment_type = payment_type
            self.extra = extra
            self.mta_tax = mta_tax
            self.tip_amount = tip_amount
            self.tolls_amount = tolls_amount
            self.improvement_surcharge = improvement_surcharge
            self.duration = duration
            self.pickup_day_of_week = pickup_day_of_week
            self.pickup_month = pickup_month


        except Exception as e:
            raise NycException(e, sys) from e

    def get_nyc_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from NycData class input
        """
        try:
            
            nyc_input_dict = self.get_nyc_data_as_dict()
            return DataFrame(nyc_input_dict)
        
        except Exception as e:
            raise NycException(e, sys) from e


    def get_nyc_data_as_dict(self):
        """
        This function returns a dictionary from NycData class input 
        """
        logging.info("Entered get_nyc_data_as_dict method as NycData class")

        try:
            input_data = {
                "vendorid": [self.vendorid],
                "passenger_count": [self.passenger_count],
                "trip_distance": [self.trip_distance],
                "ratecodeid": [self.ratecodeid],
                "pulocationid": [self.pulocationid],
                "dolocationid": [self.dolocationid],
                "payment_type": [self.payment_type],
                "extra": [self.extra],
                "mta_tax": [self.mta_tax],
                "tip_amount": [self.tip_amount],
                "tolls_amount": [self.tolls_amount],
                "improvement_surcharge": [self.improvement_surcharge],
                "duration": [self.duration],
                "pickup_hour": [self.pickup_hour],
                "pickup_day": [self.pickup_day],
                "pickup_day_of_week": [self.pickup_day_of_week],
                "pickup_month": [self.pickup_month]
            }

            logging.info("Created nyc data dict")

            logging.info("Exited get_nyc_data_as_dict method as NycData class")

            return input_data

        except Exception as e:
            raise NycException(e, sys) from e

class NycClassifier:
    def __init__(self,prediction_pipeline_config: NycTaxiTripPredictorConfig = NycTaxiTripPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise NycException(e, sys)


    def predict(self, dataframe) -> str:
        """
        This is the method of NycClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of NycClassifier class")
            model = NycEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise NycException(e, sys)