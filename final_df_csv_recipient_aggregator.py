#!/usr/bin/env python3

import os
import sys
import argparse
import csv
import json
import re
import pandas as pd
import datetime as dt
import shapely
from shapely.geometry import shape
from shapely.ops import transform
import pyproj.transformer
import pathlib
import logging
import locale
import unicodedata#from io import StringIO
from tqdm import tqdm


locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

logger = logging.getLogger('china-osn-data')

SPHERICAL_CRS="EPSG:4326"
PLANAR_CRS="EPSG:6893"
GEOJSON_PATH = 'latest/geojsons'
GEOJSON_FILE_EXTENSION = '*.geojson'
GEOPANDAS_INDEX_COLUMN = 'id'
GEOPANDAS_VALUE_COLUMN = 'Amount (Constant USD2017)'

loglevel_defs = {
    "info": logging.INFO,
    "warning": logging.WARNING,
    "debug": logging.DEBUG,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "fatal": logging.FATAL
}

def to_loglevel(x):
    return loglevel_defs[x]

headers = {
    "id": "int",
    "AidData TUFF Project ID": "int",
    # "Recommended For Aggregates": "bool",
    # "Umbrella": "bool",
    "Title": "str",
    "Status": "category",
    # "Implementation Start Year": "int",
    # "Completion Year": "int",
    "Flow Type": "category",
    "Flow Class": "category",
    "AidData Sector Name": "category",
    # "Commitment Year": "int",
    "Funding Agencies": "str",
    "Receiving Agencies": "category",
    "Implementing Agencies": "str",
    # "Recipient": "category",
    "Amount (Constant USD2017)": "float",
    # "Planned Implementation Start Date (MM/DD/YYYY)": "str",
    # "Planned Completion Date (MM/DD/YYYY)": "str",
    # "Actual Implementation Start Date (MM/DD/YYYY)": "str",
    # "Actual Completion Date (MM/DD/YYYY)": "str",
    "finance_type": "category",
    "viz_geojson_url": "str",
    "dl_geojson_url": "str"
}

datetime_columns = [
]


def to_country(x):
    if isinstance(x, str):
        return countryconvertion[x] if x in countryconvertion else unicodedata.normalize('NFD', x).encode("ascii", "ignore").decode()


def to_year(x):
    return dt.datetime.strptime(str(int(float(x))), '%Y') if x else float('NaN')


def to_datetime(x):
    return dt.datetime.fromisoformat(x) if x else float('NaN')

def to_bool(x):
    return x == 'Yes'

def to_json(o):
    if isinstance(o, dt.datetime):
        return dt.datetime.strftime(o, "%Y") if not pd.isnull(o) else None

converters = {
    "Recipient": to_country,
    "Recommended For Aggregates": to_bool,
    "Umbrella": to_bool,
    "Implementation Start Year": to_year,
    "Completion Year": to_year,
    "Commitment Year": to_year,
    "Planned Implementation Start Date (MM/DD/YYYY)": to_datetime,
    "Planned Completion Date (MM/DD/YYYY)": to_datetime,
    "Actual Implementation Start Date (MM/DD/YYYY)": to_datetime,
    "Actual Completion Date (MM/DD/YYYY)": to_datetime
}

def date_parser(date):
    return dt.datetime.fromisoformat(date) if date else float('NaN')


countryconvertion = {
    "Russia": "Russian Federation",
    "Iran": "Iran (Islamic Republic of)",
    "Micronesia": "Micronesia (Federated States of)",
    "Democratic People's Republic of Korea": "Korea (Democratic People's Republic of)",
    "Bolivia": "Bolivia (Plurinational State of)",
    "Democratic Republic of the Congo": "Congo, Democratic Republic of the",
    "Venezuela": "Venezuela (Bolivarian Republic of)",
    #"Cote d'Ivoire": "Côte d'Ivoire",
    "Curaçao": "Curacao",
    "Moldova": "Moldova, Republic of",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Tanzania": "Tanzania, United Republic of",
    "West Bank and Gaza Strip": "Palestine, State of",
    "Curacao": "Curaçao",
    "Africa, regional": float('NaN'),
    "Multi-Region": float('NaN'),
}

class CountryCode():
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self._data = json.load(f)
        for c in self._data:
            c["name"] = unicodedata.normalize('NFD', c["name"]).encode("ascii", "ignore").decode()
            #print(c["name"])

    def lookup(self, x):
        if pd.isna(x):
            return float('NaN')
        for c in self._data:
            if c["alpha-3"] == x:
                return c["name"]
        #raise KeyError("{x} not found".format(x=x))
        return float('NaN')

    def reverse_lookup(self, x):
        if pd.isna(x):
            return float('NaN')
        for c in self._data:
            if c["name"] == unicodedata.normalize('NFD', x).encode("ascii", "ignore").decode():
                return c["alpha-3"]
        raise KeyError("{x} not found".format(x=x))
        #return float('NaN')


def import_geojson(path, metric):
    tr2planar = pyproj.transformer.Transformer.from_crs(crs_from=SPHERICAL_CRS, crs_to=PLANAR_CRS, always_xy=True)
    tr2spheric = pyproj.transformer.Transformer.from_crs(crs_from=PLANAR_CRS, crs_to=SPHERICAL_CRS, always_xy=True)

    a = []

    files = pathlib.Path(path).glob(GEOJSON_FILE_EXTENSION)
    logger.info('Importing GeoJson file from path {p}'.format(p=path))

    for f in tqdm(sorted(files), desc="Loading GeoJSON", leave=False):
        logger.debug('Importing GeoJson file {f}'.format(f=f))
        with open(f, "r") as geofile:
            js = json.load(geofile)
        for x in js["features"]:
            feature = {}
            for k, v in x["properties"].items():
                if k in converters:
                    feature[k] = converters[k](v)
                else:
                    feature[k] = v
            center = transform(tr2planar.transform, shape(x["geometry"])).centroid
            geom = transform(tr2spheric.transform, center)

            feature["geometry"] = geom
            a.append(feature)

    gdf = pd.DataFrame.from_records(a, index="id")

    return parse_geojson(gdf, metric)

def default_json(o):
    if isinstance(o, dt.datetime):
        return o.isoformat()
    else:
        return o

def parse_geojson(gdf, metric):
    # To calculate center of region it's needed transforming geographical spherical coordinates
    # to planar coordinates using "to_crs()" method.
    # In this case we use Mercator projection ("epsg:3857").
    # After having calculated centroid we come back to geographical spherical coordinates again
    # with "to_crs()" method but now with "epsg:4326" projection.

    # Update to EPSG:6893 = WGS 84 / World Mercator + EGM2008 height
    # https://epsg.io/6893

    value_min = gdf[GEOPANDAS_VALUE_COLUMN].min()
    value_max = gdf[GEOPANDAS_VALUE_COLUMN].max()
    value_range = value_max - value_min
    gdf['value'] = gdf[GEOPANDAS_VALUE_COLUMN].map(lambda x: (x - value_min) / value_range)
    gdf['group'] = gdf["Flow Type"].map(lambda x: x.lower().capitalize())
    gdf['country'] = gdf['Recipient'].apply(to_country)
    gdf['filled'] = gdf["Status"] == "Completion"
    gdf['opacity'] = 0.33
    gdf['indexValue'] = gdf[metric].apply(lambda x: dt.datetime.strftime(x, "%Y") if not isinstance (x, type(pd.NaT)) else float('NaN'))

    return gdf


def df_to_geojson(df):
    # create a new python dict to contain our geojson data, using geojson format
    geojson = {'type':'FeatureCollection', 'features':[]}

    # loop through each row in the dataframe and convert each row to geojson format
    for idx, row in df.iterrows():
        # create a feature template to fill in
        feature = {'type':'Feature',
                   'properties':{}}

        # fill in the coordinates
        feature['geometry'] = shapely.geometry.mapping(row["geometry"])

        # for each column, get the value and add it as a new feature property
        for k, v in row.items():
            if k != "geometry":
                feature['properties'][k] = v

        # add this feature (aka, converted dataframe row) to the list of features inside our dict
        geojson['features'].append(feature)

    return geojson


def main(argv):
    parser = argparse.ArgumentParser(
        description = """
        Tools to aggregate final_df.csv data by recipient country.

        input file example: output_data/2.0release/results/2021_09_29_12_06/final_df.csv
        """
    )
    parser.add_argument("input", type=str, help="input file")
    parser.add_argument("--geojson-path", "-g", dest="geojson", action="store",
                        default=GEOJSON_PATH,
                        required=False, type=str, help="output filename")
    parser.add_argument("--metric", "-m", dest="metric", action="store",
                        default="Implementation Start Year",
                        required=False, type=str, help="output filename")
    parser.add_argument("--output", "-o", dest="output", action="store",
                        required=False, type=str, help="output filename")
    parser.add_argument("--iso-3166", "-i", dest="isoa3db", action="store",
                        required=False, type=str, help="iso alpha-3 country code database json file")
    parser.add_argument("--loglevel", "-l", dest="loglevel", action="store",
                        default="info", type=str,
                        choices=loglevel_defs.keys(), help="log level")
    parser.add_argument("--quiet", "-q", dest="quiet", action="store_true",
                        help="quiet execution")

    args = parser.parse_args(argv)

    loglevel = to_loglevel(args.loglevel)
    # logging.basicConfig(level=logging.NOTSET)
    logger.setLevel(loglevel)

    input_ = args.input
    output_ = args.output
    isoa3db = args.isoa3db if args.isoa3db else 'iso_a3.json'

    #metric = "Commitment Year"
    metric = args.metric

    countrycode = CountryCode(isoa3db)

    dataout = {
        "type": "d3.js:geo",
        "locale": "en-US",
        "format": {
            "style": "currency",
            "currency": "USD",
            "notation": "compact",
            "minimumFractionDigits": 0,
            "maximumFractionDigits": 2
        },
        # "data": {}
    }
    # datain = dataout["data"]

    with open(input_, 'r') as csvfile:
        df = pd.read_csv(
            csvfile,
            dtype=headers, converters=converters,
            parse_dates=datetime_columns, date_parser=date_parser
        )


    df["Recipient Code"] = df['Recipient'].map(countrycode.reverse_lookup)
    df["Recipient Code"].astype("category")

    df = df[df["Recipient"].isna() == False]
    df = df.sort_values(["Recipient Code", metric], ascending=[True, True])

    new_g = df.groupby(by=["Recipient Code", metric])["Amount (Constant USD2017)"].sum().reset_index()
    new_g[metric] = new_g[metric].map(lambda x: dt.datetime.strftime(x, "%Y"))
    new_g = new_g.pivot(columns="Recipient Code", index=metric, values="Amount (Constant USD2017)").fillna(0)
    new_g = new_g.cumsum(axis=0)
    new_g_dict = new_g.to_dict(orient='split')
    new_g_dict["key"] = new_g_dict["columns"]
    del new_g_dict["columns"]
    dataout["dataset"] = new_g_dict

    dataout["dataset"]["label"] = {"data": "Amount of investments (USD2017)"}

    gdf = import_geojson(args.geojson, metric)
    gdf["country"] = gdf['country'].map(to_country)
    gdf["country3ISO"] = gdf['country'].map(countrycode.reverse_lookup)
    gdf["country3ISO"].astype("category")
    gdf = gdf.loc[gdf[metric].notnull()]
    gdf = gdf.replace([float('NaN')], [None])

    dataout["geojson"] = df_to_geojson(gdf)

    if output_:
        with open(output_, 'w') as out:
            json.dump(dataout, out, default=default_json)
    else:
        print(json.dumps(dataout, default=default_json))


if __name__ == "__main__":
    main(sys.argv[1:])
