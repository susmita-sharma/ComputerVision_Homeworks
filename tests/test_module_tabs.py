import re

import pytest

import main

PAGES = [
    ("/hw2/calibrate", "Run", "/hw2/sample-output?part=calibration"),
    ("/hw2/measure", "Run", "/hw2/sample-output?part=measurement"),
    ("/hw2/validate", "Run", "/hw2/sample-output?part=validation"),
    ("/hw3/", "Run", "/hw3/sample-output"),
    ("/hw4/", "Run", "/hw4/sample-output"),
    ("/hw5/", "Overview", "/hw5/sample-output"),
    ("/hw5/flow", "Run", "/hw5/sample-output?part=flow"),
    ("/hw5/tracking", "Run", "/hw5/sample-output?part=tracking"),
    ("/hw5/sfm", "Run", "/hw5/sample-output?part=sfm"),
    ("/hw7/", "Overview", "/hw7/sample-output"),
    ("/hw7/stereo", "Run", "/hw7/sample-output?part=stereo"),
    ("/hw7/room", "Run", "/hw7/sample-output?part=room"),
]
SAMPLES = [(s, p) for p, label, s in PAGES if label == "Run"]


def tabs(url):
    response = main.app.test_client().get(url)
    assert response.status_code == 200
    bar = re.search(r'<div class="module-tabs">(.*?)</div>', response.data.decode(), re.S).group(1)
    return re.findall(r'<a href="([^"]+)"\s*(class="active")?>([^<]+)</a>', bar)


@pytest.mark.parametrize("url,label,sample_url", PAGES)
def test_module_page_links_to_its_own_sample(url, label, sample_url):
    assert tabs(url) == [(url, 'class="active"', label), (sample_url, "", "Sample Output")]


@pytest.mark.parametrize("sample_url,run_url", SAMPLES)
def test_sample_page_links_back_to_its_module(sample_url, run_url):
    run, sample = tabs(sample_url)
    assert run == (run_url, "", "Run")
    assert sample[1:] == ('class="active"', "Sample Output")
    assert sample[0].rstrip("?") == sample_url


def test_no_tabs_off_module_pages():
    client = main.app.test_client()
    for url in ("/", "/challenge1/"):
        assert 'class="module-tabs"' not in client.get(url).data.decode()
