import json
import os
import requests


def test_json_request():
    data = ["great", "fantastic", "movie"]

    serialized_output = requests.post("http://localhost:8080/invocations",
                                      data=json.dumps(data),
                                      headers={'Content-type': 'application/json',
                                               'Accept': 'application/json'}).content

    prediction_result = json.loads(serialized_output)
    classes = prediction_result['result']['classifications'][0]['classes']
    assert len(classes) == 2
    assert classes[0].keys() == [u'score', u'label']


def test_assets_were_restored():
    found_vocabulary_files = _find_files("/opt/ml/model/export/Servo",
                                         dir_predicate=lambda d: d.endswith("/assets"),
                                         file_predicate=lambda f: f == "vocabulary.txt")
    assert len(found_vocabulary_files) > 0, 'At least one "vocabulary.txt" asset file is expected'


def _find_files(root_dir, dir_predicate=lambda _: True, file_predicate=lambda _: True):
    found_files = []
    for root, _, files in os.walk(root_dir):
        if dir_predicate(root):
            found_files += [root + "/" + file for file in files if file_predicate(file)]
    return found_files
