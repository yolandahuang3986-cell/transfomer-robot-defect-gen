import json
from pathlib import Path
from jsonschema import Draft202012Validator

SCHEMA = Path(__file__).resolve().parents[2] / 'schemas/result.schema.json'


def validate_result(result):
    # JSON Schema treats some non-finite floats as numbers: reject them first.
    json.dumps(result, allow_nan=False)
    Draft202012Validator(json.loads(SCHEMA.read_text())).validate(result)
    return result


def save_result(path, result):
    validate_result(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
