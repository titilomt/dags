import json
import jsonschema
from jsonschema import validate


def validate_input(json_data):
    schema = {
        # "type" : "object",
        "properties": {
            "sepal_length": {"type": "number"},
            "sepal_width": {"type": "number"},
            "petal_length": {"type": "number"},
            "petal_width": {"type": "number"},
        },
    }

    validate(instance=json_data, schema=schema)
