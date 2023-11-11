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
GEOPANDAS_VALUE_COLUMN = 'Amount (Constant USD 2021)'

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

header_parser_rules = [
    (re.compile(r'[()]').sub, r''),
    (re.compile(r'\-').sub, r'_'),
    (re.compile(r' ').sub, r'_'),
]

gcdf30_headers = {
    "AidData Record ID": "int64",
    "Recipient ISO-3": "category",
    "Recipient Region": "category",
    "Title": "str",
    "Description": "str",
    "Staff Comments": "str",
    "Status": "category",
    "Intent": "category",
    "Flow Type": "category",
    "Flow Type Simplified": "category",
    "OECD ODA Concessionality Threshold": "float64",
    "Flow Class": "category",
    "Sector Code": "int64",
    "Sector Name": "category",
    "Funding Agencies": "category",
    "Funding Agencies Type": "category",
    "Collateral": "str",
    "Amount (Original Currency)": "float64",
    "Original Currency": "category",
    "Amount (Constant USD 2021) ": "float64",
    "Amount (Nominal USD) ": "float64",
    "Adjusted Amount (Original Currency) ": "float64",
    "Adjusted Amount (Constant USD 2021) ": "float64",
    "Adjusted Amount (Nominal USD) ": "float64",
    "Deviation from Planned Implementation Start Date": "float64",
    "Deviation from Planned Completion Date": "float64",
    "Maturity": "float64",
    "Interest Rate": "float64",
    "Grace Period": "float64",
    "Management Fee": "float64",
    "Commitment Fee": "float64",
    "Insurance Fee (Percent)": "float64",
    "Insurance Fee (Nominal USD)": "float64",
    "Default Interest Rate": "float64",
    "Grant Element (OECD Cash-Flow)": "float64",
    "Grant Element (OECD Grant-Equiv)": "float64",
    "Grant Element (IMF)": "float64",
    "Number of Lenders": "category",
    "Project JV/SPV Host Government Ownership": "category",
    "Project JV/SPV Chinese Government Ownership": "category",
    "Level of Public Liability": "category",
    "Total Source Count": "float64",
    "Official Source Count": "float64",
    "Source URLs": "str",
    "Source Titles": "str",
    "OECD ODA Income Group": "category",
    "Location Narrative": "str",
    "Geographic Level of Precision Available": "category",
    "Source Quality Score": "float64",
    "Data Completeness Score": "float64",
    "Implementation Detail Score": "float64",
    "Loan Detail Score": "float64",    
}

gcdf30_datetime_columns = [
]

gcdf30_adm_headers = {
    "id": "int64",
    "shapeID": "str",
    "shapeGroup": "str",
    "shapeName": "str",
    "intersection_ratio": "float64",
    "even_split_ratio": "float64",
    "intersection_ratio_commitment_value": "float64",
    "even_split_ratio_commitment_value": "float64",
    "centroid_longitude": "float64",
    "centroid_latitude": "float64",
}

def to_country(x):
    if isinstance(x, str):
        return countryconvertion[x] if x in countryconvertion else unicodedata.normalize('NFD', x).encode("ascii", "ignore").decode()


def to_year(x):
    return dt.datetime.strptime(str(int(float(x))), '%Y') if x else float('NaN')


def to_datetime(x):
    if isinstance(x, dt.datetime):
        return x
    elif isinstance(x, str) and len(x) > 0:
        return dt.datetime.fromisoformat(x)
    else:
        return float('NaN')


def to_list(x):
    return tuple(re.split('|', str(x)))


def to_bool(x):
    return x == 'Yes'


def to_json(o):
    if isinstance(o, dt.datetime):
        return dt.datetime.strftime(o, "%Y") if not pd.isnull(o) else None

gcdf30_converters = {
    "Recommended For Aggregates": to_bool,
    "AidData Parent ID": to_list,
    "Umbrella": to_bool,
    "Financier Country": to_country,
    "Recipient": to_country,
    "Commitment Year": to_year,
    "Implementation Start Year": to_year,
    "Completion Year": to_year,
    "Infrastructure": to_bool,
    "COVID": to_bool,
    "Co-financed": to_bool,
    "Co-financing Agencies": to_list,
    "Co-financing Agencies Type": to_list,
    "Direct Receiving Agencies": to_list,
    "Direct Receiving Agencies Type": to_list,
    "Indirect Receiving Agencies": to_list,
    "Indirect Receiving Agencies Type": to_list,
    "On-lending": to_bool,
    "Implementing Agencies": to_list,
    "Implementing Agencies Type": to_list,
    "Guarantee Provided": to_bool,
    "Guarantor": to_list,
    "Guarantor Agency Type": to_list,
    "Insurance Provided": to_bool,
    "Insurance Provider": to_list,
    "Insurance Provider Agency Type": to_list,
    "Collateralized": to_bool,
    "Collateral Provider": to_list,
    "Collateral Provider Agency Type": to_list,
    "Security Agent/Collateral Agent": to_list,
    "Security Agent/Collateral Agent Type": to_list,
    "Amount Estimated": to_bool,
    "Financial Distress": to_bool,
    "Commitment Date (MM/DD/YYYY)": to_datetime,
    "Commitment Date Estimated": to_bool,
    "Planned Implementation Start Date (MM/DD/YYYY)": to_datetime,
    "Actual Implementation Start Date (MM/DD/YYYY)": to_datetime,
    "Actual Implementation Start Date Estimated": to_bool,
    "Planned Completion Date (MM/DD/YYYY)": to_datetime,
    "Actual Completion Date (MM/DD/YYYY)": to_datetime,
    "Actual Completion Date Estimated": to_bool,
    "First Loan Repayment Date": to_datetime,
    "Last Loan Repayment Date": to_datetime,
    "Export Buyer's Credit": to_bool,
    "Supplier’s Credit/Export Seller’s Credit": to_bool,
    "Interest-Free Loan": to_bool,
    "Refinancing": to_bool,
    "Investment Project Loan": to_bool,
    "M&A": to_bool,
    "Working Capital": to_bool,
    "EPCF": to_bool,
    "Lease": to_bool,
    "FXSL/BOP": to_bool,
    "CC IRS": to_bool,
    "RCF": to_bool,
    "GCL": to_bool,
    "PBC": to_bool,
    "PxF/Commodity Prepayment": to_bool,
    "Inter-Bank Loan": to_bool,
    "Overseas Project Contracting Loan": to_bool,
    "DPA": to_bool,
    "Project Finance": to_bool,
    "Involving Multilateral": to_bool,
    "Non-Chinese Financier": to_bool,
    "Short-Term": to_bool,
    "Rescue": to_bool,
    "Source Publishers": to_list,
    "Source Resource Types": to_list,
    "Contact Name": to_list,
    "Contact Position": to_list,
    "ODA Eligible Recipient": to_bool,
    "ADM1 Level Available": to_bool,
    "ADM2 Level Available": to_bool,
    "Geospatial Feature Available": to_bool,
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
    "Cote d\"Ivoire": "Côte d'Ivoire",
    "Laos": "Lao People's Democratic Republic",
    "Brunei": "Brunei Darussalam",
    "Syria": "Syrian Arab Republic",
    "North Korea": "Korea (Democratic People's Republic of)",
    "Vietnam" :"Viet Nam",
    "DR Congo": "Congo, Democratic Republic of the",
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
            return [float('NaN')]
        for c in self._data:
            if c["alpha-3"] == x:
                return c["name"]
        #raise KeyError("{x} not found".format(x=x))
        return float('NaN')

    def reverse_lookup(self, x):
        if pd.isna(x):
            return [float('NaN')]
        for c in self._data:
            if c["name"] == unicodedata.normalize('NFD', x).encode("ascii", "ignore").decode():
                return c["alpha-3"]
        raise KeyError("{x} not found".format(x=x))
        #return float('NaN')


def row_to_geojson(x, aggr_by, group_by, value, filled, opacity):
    feature = {'type': 'Feature', 'properties': {}}
    feature['properties'][ 'opacity' ] = opacity
    feature['properties'][ 'indexValue' ] = dt.datetime.strftime( x[ aggr_by ], "%Y") if not isinstance(x[ aggr_by ], type(pd.NaT)) else float('NaN')
    feature['properties'][ 'group' ] = x[ group_by ]
    feature['properties'][ 'filled' ] = x[ filled['by'] ] == filled['value']
    feature['properties'][ 'value' ] = ( x[ value['name'] ] - value['min'] ) / ( value['max'] - value['min'] )
    feature['geometry'] = {
        'type': 'Point',
        'coordinates': [
            x['centroid_longitude'],
            x['centroid_latitude'],
        ],
    }

    return feature


def column_name_parser(x):
    t = x
    for r in header_parser_rules:
        t = r[0](r[1], t)
    t = t.lower()
    return t


def default_json(o):
    if isinstance(o, dt.datetime):
        return o.isoformat()
    else:
        return o


def main(argv):
    parser = argparse.ArgumentParser(
        description = """
        Tools to aggregate final_df.csv data by recipient country.

        input file example: output_data/2.0release/results/2021_09_29_12_06/final_df.csv
        """
    )
    parser.add_argument("input", type=str, help="input file")
    parser.add_argument("--adm-loc-files", "-a", dest="adm_loc_files", action="store", nargs="+",
                        required=True, type=str, help="adm location files")
    parser.add_argument("--aggr-by", "-m", dest="aggr_by", action="store",
                        default="Commitment Year",
                        required=False, type=str, help="column to aggregate data for index")
    parser.add_argument("--group-by", "-g", dest="group_by", action="store",
                        default="Flow Type",
                        required=False, type=str, help="column to categorize data")
    parser.add_argument("--fill-by", "-f", dest="fill_by", action="store",
                        nargs=2, default=["Status", "Completion"],
                        required=False, type=str, help="column that defined if data geojson point should be filled "
                        "(default:  'Status' 'Completion')")
    parser.add_argument("--value-column", "-v", dest="value_column", action="store",
                        default=GEOPANDAS_VALUE_COLUMN,
                        required=False, type=str, help="Metric to index aggregated data")
    parser.add_argument("--sheet", "-s", dest="sheet", action="store",
                        default="GCDF_3.0",
                        required=False, type=str, help="excel sheet")
    parser.add_argument("--output", "-o", dest="output", action="store",
                        required=False, type=str, help="output filename")
    parser.add_argument("--iso-3166", "-i", dest="isoa3db", action="store", default="iso_a3.json",
                        required=False, type=str, help="iso alpha-3 country code database json file")
    parser.add_argument("--loglevel", "-l", dest="loglevel", action="store",
                        default="info", type=str,
                        choices=loglevel_defs.keys(), help="log level")
    #parser.add_argument("--quiet", "-q", dest="quiet", action="store_true",
    #                    help="quiet execution")

    args = parser.parse_args(argv)

    loglevel = to_loglevel(args.loglevel)
    # logging.basicConfig(level=logging.NOTSET)
    logger.setLevel(loglevel)

    input_ = args.input
    output_ = args.output
    isoa3db = args.isoa3db

    aggr_by = column_name_parser(args.aggr_by)
    group_by = column_name_parser(args.group_by)
    fill_by = { "by": column_name_parser(args.fill_by[0]),
                "value": args.fill_by[1] }
    value_column = column_name_parser(args.value_column)

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
        "command": command,
        # "data": {}
    }
    # datain = dataout["data"]

    df = pd.read_excel(
        input_, sheet_name=args.sheet,
        dtype=gcdf30_headers, converters=gcdf30_converters,
        parse_dates=gcdf30_datetime_columns, date_parser=date_parser
    )

    df = df.rename(columns=column_name_parser)

    geojson_dfs = []
    for __loc_filename in tqdm(args.adm_loc_files, desc="Processing ADM location files", leave=False):
        geojson_df = pd.read_csv(__loc_filename, dtype=gcdf30_adm_headers,
                                 converters=gcdf30_converters)
        geojson_df = geojson_df.merge(df, left_on="id", right_on="aiddata_record_id")
        geojson_df = geojson_df[ geojson_df[value_column].isna() == False ]
        geojson_value = {
            "name": value_column,
            "min": geojson_df[ value_column ].min(),
            "max": geojson_df[ value_column ].max(),
        }
        geojson_df["feature"] = geojson_df.apply(lambda x: row_to_geojson(x, aggr_by, group_by,
                                                                          geojson_value, fill_by,
                                                                          0.33), axis=1)
        geojson_dfs.append( geojson_df )

    df = df[ df["recipient_iso_3"].notnull() ]
    df["recipient_code"] = df['recipient'].map(countrycode.reverse_lookup)
    df["recipient_code"].astype("category")

    #df = df[df["Recipient"].isna() == False]
    df = df.sort_values(["recipient_code", aggr_by], ascending=[True, True])

    new_g = df.groupby(by=["recipient_code", aggr_by], as_index=False)[value_column].sum()
    new_g[aggr_by] = new_g[aggr_by].map(lambda x: dt.datetime.strftime(x, "%Y"))
    new_g = new_g.pivot(columns="recipient_code", index=aggr_by, values=value_column).fillna(0)
    new_g = new_g.cumsum(axis=0)
    new_g_dict = new_g.to_dict(orient='split')
    new_g_dict["key"] = new_g_dict["columns"]
    del new_g_dict["columns"]
    dataout["dataset"] = new_g_dict

    dataout["dataset"]["label"] = {"data": "Total of " + args.value_column}

    # gdf = import_geojson(args.geojson, aggr_by)
    # gdf["country"] = gdf['country'].map(to_country)
    # gdf["country3ISO"] = gdf['country'].map(countrycode.reverse_lookup)
    # gdf["country3ISO"].astype("category")
    # gdf = gdf.loc[gdf[aggr_by].notnull()]
    # gdf = gdf.replace([float('NaN')], [None])
    #dataout["geojson"] = df_to_geojson(gdf)

    dataout["geojson"] = { "type": "FeatureCollection",
                        "features": [] }
    for geojson in geojson_dfs:
        dataout["geojson"]["features"].extend( geojson["feature"].tolist() )

    if output_:
        with open(output_, 'w') as out:
            json.dump(dataout, out, default=default_json)
    else:
        print(json.dumps(dataout, default=default_json))


if __name__ == "__main__":
    command = " ".join(sys.argv)
    main(sys.argv[1:])
