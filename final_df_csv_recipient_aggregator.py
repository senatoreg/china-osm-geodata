#!/usr/bin/env python3

import sys
import argparse
import csv
import json
import re

countryconvertion = {
    "Russia": "Russian Federation",
    "Iran": "Iran (Islamic Republic of)",
    "Micronesia": "Micronesia (Federated States of)",
    "Democratic People's Republic of Korea": "Korea (Democratic People's Republic of)",
    "Bolivia": "Bolivia (Plurinational State of)",
    "Democratic Republic of the Congo": "Congo, Democratic Republic of the",
    "Venezuela": "Venezuela (Bolivarian Republic of)",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Moldova": "Moldova, Republic of",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Tanzania": "Tanzania, United Republic of",
    "West Bank and Gaza Strip": "Palestine, State of",
    "Curacao": "Curaçao",
    "Africa, regional": None,
    "Multi-Region": None,
}

def main(argv):
    parser = argparse.ArgumentParser(
        description = """
        Tools to aggregate final_df.csv data by recipient country.

        input file example: output_data/2.0release/results/2021_09_29_12_06/final_df.csv
        """
    )
    parser.add_argument("input", type=str, nargs=1, help="input file")
    parser.add_argument("--output", dest="output", action="store",
                        required=False, type=str, help="output filename")
    parser.add_argument("--iso-a3-db", dest="isoa3db", action="store",
                        required=False, type=str, help="iso alpha-3 country code database json file")

    args = parser.parse_args(argv)

    input_ = args.input[0]
    output_ = args.output
    isoa3db = args.isoa3db if args.isoa3db else 'iso_a3.json'

    with open(isoa3db, 'r') as isoa3file:
        isoa3 = json.load(isoa3file)

    dataout = {"locale": "en-US",
               "format": {
                   "style": "currency",
                   "currency": "USD",
                   "notation": "compact",
                   "minimumFractionDigits": 0,
                   "maximumFractionDigits": 2
               },
               "data": {}
               }
    datain = dataout["data"]

    with open(input_, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            recipient = row['Recipient']
            code = None

            if recipient in countryconvertion:
                recipient = countryconvertion[recipient]

            if not recipient:
                continue

            for country in isoa3:
                if country["name"] == recipient:
                    code = country["alpha-3"]

            if not code:
                print(recipient + ": code not found!", file=sys.stderr)
                continue

            if code not in datain:
                datain[code] = 0

            amount = float(row['Amount (Constant USD2017)']) if row['Amount (Constant USD2017)'] else 0
            datain[code] += amount

    if output_:
        with open(output_, 'w') as out:
            json.dump(dataout, out)
    else:
        print(json.dumps(dataout))


if __name__ == "__main__":
    main(sys.argv[1:])
