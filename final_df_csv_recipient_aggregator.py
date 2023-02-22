#!/usr/bin/env python3

import os
import sys
import argparse
import csv
import json
import re
import pandas as pd
import numpy as np
import datetime as dt
import pandas as pd
from pandas.io.formats.printing import _pprint_dict
import shapely as shp
import geojson
import fiona
import geopandas as gpd
import pathlib
import logging
import locale
import unicodedata
from io import StringIO
from tqdm import tqdm


locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

logger = logging.getLogger('china-osn-data')

GEOJSON_PATH = 'latest/geojsons'
TEMP_PATH = '/tmp'
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
    "id": "category",
    "AidData TUFF Project ID": "category",
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
    return dt.datetime.strptime(str(int(float(x))), '%Y') if x else np.nan


def to_datetime(x):
    return dt.datetime.fromisoformat(x) if x else np.nan

def to_bool(x):
    return x == 'Yes'

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
    return dt.datetime.fromisoformat(date) if date else np.nan


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
    "Africa, regional": np.nan,
    "Multi-Region": np.nan,
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
            return np.nan
        for c in self._data:
            if c["alpha-3"] == x:
                return c["name"]
        #raise KeyError("{x} not found".format(x=x))
        return np.nan

    def reverse_lookup(self, x):
        if pd.isna(x):
            return np.nan
        for c in self._data:
            if c["name"] == unicodedata.normalize('NFD', x).encode("ascii", "ignore").decode():
                return c["alpha-3"]
        raise KeyError("{x} not found".format(x=x))
        #return np.nan


def import_geojson(path, quiet=False):
    gdf = gpd.GeoDataFrame()
    files = pathlib.Path(path).glob(GEOJSON_FILE_EXTENSION)
    logger.info('Importing GeoJson file from path {p}'.format(p=path))
    if quiet:
        w = sorted(files)
    else:
        w = tqdm(sorted(files), desc="Loading GeoJSON", leave=False)
    for f in w:
        logger.debug('Importing GeoJson file {f}'.format(f=f))
        geofile = gpd.read_file(f)
        geofile = geofile.set_index([GEOPANDAS_INDEX_COLUMN])
        gdf = gdf.append(geofile)

    return parse_geojson(gdf)

def to_json(o):
    if isinstance(o, dt.datetime):
        return dt.datetime.strftime(o, "%Y") if not pd.isnull(o) else None

def parse_geojson(gdf):
    # To calculate center of region it's needed transforming geographical spherical coordinates
    # to planar coordinates using "to_crs()" method.
    # In this case we use Mercator projection ("epsg:3857").
    # After having calculated centroid we come back to geographical spherical coordinates again
    # with "to_crs()" method but now with "epsg:4326" projection.

    # Update to EPSG:6893 = WGS 84 / World Mercator + EGM2008 height
    # https://epsg.io/6893

    new_gdf = gpd.GeoDataFrame(index=gdf.index.tolist())
    new_gdf['geometry'] = gdf.to_crs('epsg:6893').centroid.to_crs('epsg:4326')
    gdf.set_geometry('geometry')

    value_min = gdf[GEOPANDAS_VALUE_COLUMN].min()
    value_max = gdf[GEOPANDAS_VALUE_COLUMN].max()
    value_range = value_max - value_min
    new_gdf['value'] = gdf[GEOPANDAS_VALUE_COLUMN].map(lambda x: (x - value_min) / value_range)
    new_gdf['group'] = gdf["Flow Type"].map(lambda x: x.lower().capitalize())
    new_gdf['country'] = gdf['Recipient'].apply(to_country)
    new_gdf['filled'] = gdf["Status"] == "Completion"
    new_gdf['opacity'] = 0.33
    new_gdf['indexValue'] = gdf["Commitment Year"].apply(lambda x: str(x))

    return new_gdf

def main(argv):
    parser = argparse.ArgumentParser(
        description = """
        Tools to aggregate final_df.csv data by recipient country.

        input file example: output_data/2.0release/results/2021_09_29_12_06/final_df.csv
        """
    )
    parser.add_argument("input", type=str, nargs='?', help="input file")
    parser.add_argument("--source-path", "-s", dest="source", action="store",
                        default=GEOJSON_PATH,
                        required=False, type=str, help="output filename")
    parser.add_argument("--temp-path", "-t", dest="tmp_path", action="store",
                        default=TEMP_PATH,
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
    df = df.sort_values(["Recipient Code", "Commitment Year"], ascending=[True, True])


    new_g = df.groupby(by=["Recipient Code", "Commitment Year"])["Amount (Constant USD2017)"].sum().reset_index()
    new_g["Commitment Year"] = new_g["Commitment Year"].map(lambda x: dt.datetime.strftime(x, "%Y"))
    new_g = new_g.pivot(columns="Recipient Code", index="Commitment Year", values="Amount (Constant USD2017)").fillna(0)
    new_g = new_g.cumsum(axis=0)
    new_g_dict = new_g.to_dict(orient='split')
    new_g_dict["key"] = new_g_dict["columns"]
    del new_g_dict["columns"]
    dataout["dataset"] = new_g_dict

    dataout["dataset"]["label"] = {"data": "Amount of investments"}

    # c_g = df.groupby(by=["Recipient Code"])
    # # dataout["data"] = c_g["Amount (Constant USD2017)"].sum().sort_values(ascending=[False]).to_dict()
    # values = c_g["Amount (Constant USD2017)"].sum().sort_values(ascending=[False]).to_dict()
    # for k, v in values.items():
    #     datain[k] = {"value": v, "label": "Amount of investments"}

    gdf = import_geojson(args.source, args.quiet)
    gdf["country"] = gdf['country'].map(to_country)
    gdf["country3ISO"] = gdf['country'].map(countrycode.reverse_lookup)
    gdf["country3ISO"].astype("category")

    # save data in temporary directory
    tmp_file = args.tmp_path + '/' + 'china-osm-data' + str(os.getpid()) + '.geojson'
    gdf.to_file(tmp_file, driver='GeoJSON')
    with open(tmp_file, "r") as f:
        geojson_data = json.load(f)

    if loglevel >= logging.DEBUG and os.path.exists(tmp_file):
        os.remove(tmp_file)
    dataout["geojson"] = geojson_data

    # y_g = df.groupby(["Completion Year"])
    #
    # new_d = {}
    # for k, v in y_g["Amount (Constant USD2017)"].sum().to_dict().items():
    #     new_d[k.strftime('%Y')] = {"amount": v, "projects": None}
    # for k, v in y_g["id"].count().to_dict().items():
    #     year = k.strftime('%Y')
    #     if year in new_d:
    #         new_d[year]["projects"] = v
    #     else:
    #         new_d[year] = {"amount": None, "projects": v}
    #
    #
    # print(json.dumps(new_d))

    if output_:
        with open(output_, 'w') as out:
            json.dump(dataout, out)
    else:
        print(json.dumps(dataout))


if __name__ == "__main__":
    main(sys.argv[1:])
