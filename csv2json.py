#!/usr/bin/python3
import sys
import csv 
import json 
import argparse


def csv_to_json(file):
    js = []
      
    #read csv file
    with open(file, encoding='utf-8') as f: 
        #load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(f) 

        #convert each csv row into python dict
        for row in csvReader: 
            #add this python dict to json array
            js.append(row)

    return js

def main(argv):
    parser = argparse.ArgumentParser(
        description = """
        """
    )
    parser.add_argument("source", type=str, help="input file")
    parser.add_argument("--output", "-o", type=str, help="output file")

    args = parser.parse_args(argv)

    js = csv_to_json(args.source)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(js, f)
    else:
        print(json.dumps(js))

if __name__ == "__main__":
    main(sys.argv[1:])
