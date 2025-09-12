import pandas as pd
import requests
import requests_cache
import openmeteo_requests
from retry_requests import retry
import io
import json

def get_pvgis_hourly(parameters: dict):
    """
    Fetch the data from PVGIS
    :param parameters: The parameters of the PV system, specified by the user
    :return: a dataframe holding the hourly PVGIS data
    """

    URL = 'https://re.jrc.ec.europa.eu/api/v5_2/seriescalc'

    data_request = requests.get(URL,
                                params=parameters,
                                timeout=120)

    if not data_request.ok:
        try:
            err_msg = data_request.json()
        except Exception:
            data_request.raise_for_status()
        else:
            raise requests.HTTPError(err_msg['message'])

    filename = io.StringIO(data_request.text)
    try:
        src = json.load(filename)
    except AttributeError:  # str/path has no .read() attribute
        with open(str(filename), 'r') as fbuf:
            src = json.load(fbuf)

    data = pd.DataFrame(src['outputs']['hourly'])
    data.index = pd.to_datetime(data['time'], format='%Y%m%d:%H%M', utc=True)
    data = data.drop('time', axis=1)
    data = data.astype(dtype={'Int': 'int'})

    return data


def get_open_meteo_hourly(parameters):

    URL = "https://archive-api.open-meteo.com/v1/archive"

    # Set up the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    open_meteo = openmeteo_requests.Client(session=retry_session)

    responses = open_meteo.weather_api(URL, params=parameters)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_relative_humidity_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_direct_radiation = hourly.Variables(1).ValuesAsNumpy()
    hourly_diffuse_radiation = hourly.Variables(2).ValuesAsNumpy()
    date = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left")

    hourly_data = {"date": date,
                    "relative_humidity_2m": hourly_relative_humidity_2m,
                    "direct_radiation": hourly_direct_radiation,
                    "diffuse_radiation": hourly_diffuse_radiation}

    hourly_dataframe = pd.DataFrame(data=hourly_data)

    return hourly_dataframe