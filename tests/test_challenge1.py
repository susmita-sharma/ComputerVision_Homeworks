import main
from Homework.challenge1 import notebook


def test_results_are_read_from_the_notebook():
    results, figure = notebook.load()
    assert figure.startswith(b"\x89PNG")
    assert len(results["rows"]) == 8
    assert [h["name"] for h in results["homes"]] == ["IMG_2133.MOV", "IMG_2132.MOV"]
    assert results["homes"][1]["plan"][0] == "grasp the bottle"
    assert "descend(344, 738, 0.47)" in results["homes"][1]["actions"]


def test_summary_matches_the_notebook_summary_cell():
    summary = {s["method"]: s for s in notebook.load()[0]["summary"]}
    ours = summary["context-aware (ours)"]
    assert round(ours["grasp"], 2) == 0.5 and round(ours["iou"], 2) == 0.43
    assert round(ours["dim"]["iou"], 2) == 0.42 and round(ours["normal"]["iou"], 2) == 0.43
    assert summary["baseline (no context)"]["grasp"] == 0


def test_pages_and_landing_tiles():
    client = main.app.test_client()
    page = client.get("/challenge1/").data.decode()
    assert "Sample Output" in page and "IMG_2132.MOV" in page
    assert client.get("/challenge1/figure.png").mimetype == "image/png"
    home = client.get("/").data.decode()
    for link in ("/hw2/calibrate", "/hw3/", "/hw4/", "/hw5/", "/hw7/", "/challenge1/"):
        assert f'class="hub-tile" href="{link}"' in home
        assert client.get(link).status_code == 200
