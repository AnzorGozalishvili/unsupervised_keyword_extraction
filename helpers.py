import json


def read_json(path):
    with open(path) as file:
        return json.load(file)
