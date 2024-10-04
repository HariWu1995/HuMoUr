import io
import yaml
import json

from pathlib import Path
from typing import Union


def load_config(path: Union[str, Path]):
    if not isinstance(path, str):
        path = str(path)

    with open(path, 'r') as f:
        if path.lower().endswith(('.yaml','.yml')):
            config = yaml.load(f, Loader=yaml.SafeLoader)
        elif path.lower().endswith('.json'):
            config = json.load(f)
    return config


def load_multipart_file(file_content):
    file_content = file_content.file.read()
    file_content = file_content.decode()
    file_content = io.StringIO(file_content)
    return file_content


