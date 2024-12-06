

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from nyc_taxi_trips.constants import APP_HOST, APP_PORT
from nyc_taxi_trips.pipeline.prediction_pipeline import NycData, NycClassifier
from nyc_taxi_trips.pipeline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.vendorid: Optional[float] = None
        self.passenger_count: Optional[float] = None
        self.trip_distance: Optional[float] = None
        self.duration: Optional[float] = None
        self.pickup_hour: Optional[float] = None
        self.pickup_day: Optional[float] = None
        self.pickup_day_of_week: Optional[float] = None
        self.pickup_month: Optional[float] = None
        self.ratecodeid: Optional[float] = None
        self.pulocationid: Optional[float] = None
        self.dolocationid: Optional[float] = None
        self.payment_type: Optional[float] = None
        self.extra: Optional[float] = None
        self.mta_tax: Optional[float] = None
        self.tip_amount: Optional[float] = None
        self.tolls_amount: Optional[float] = None
        self.improvement_surcharge: Optional[float] = None
        

    async def get_nyc_data(self):
        form = await self.request.form()
        self.vendorid = form.get("vendorid")
        self.passenger_count = form.get("passenger_count")
        self.trip_distance = form.get("trip_distance")
        self.duration = form.get("duration")
        self.pickup_hour = form.get("pickup_hour")
        self.pickup_day = form.get("pickup_day")
        self.pickup_day_of_week = form.get("pickup_day_of_week")
        self.pickup_month = form.get("pickup_month")
        self.ratecodeid = form.get("ratecodeid")
        self.pulocationid = form.get("pulocationid")
        self.dolocationid = form.get("dolocationid")
        self.payment_type = form.get("payment_type")
        self.extra = form.get("extra")
        self.mta_tax = form.get("mta_tax")
        self.tip_amount = form.get("tip_amount")
        self.tolls_amount = form.get("tolls_amount")
        self.improvement_surcharge = form.get("improvement_surcharge")

@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "nyc.html",{"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_nyc_data()
        
        nyc_data = NycData(
                                vendorid=form.vendorid,
                                passenger_count=form.passenger_count,
                                trip_distance=form.trip_distance,
                                duration= form.duration,
                                pickup_hour= form.pickup_hour,
                                pickup_day= form.pickup_day,
                                pickup_day_of_week= form.pickup_day_of_week,
                                pickup_month= form.pickup_month,
                                ratecodeid= form.ratecodeid,
                                pulocationid= form.pulocationid,
                                dolocationid= form.dolocationid,
                                payment_type=form.payment_type,
                                extra= form.extra,
                                mta_tax= form.mta_tax,
                                tip_amount= form.tip_amount,
                                tolls_amount= form.tolls_amount,
                                improvement_surcharge= form.improvement_surcharge
                                )
        
        nyc_df = nyc_data.get_nyc_input_data_frame()

        model_predictor = NycClassifier()

        value = model_predictor.predict(dataframe=nyc_df)[0]

        status = f"The fare for this trip is {value}"
    

        return templates.TemplateResponse(
            "nyc.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
