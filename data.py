import zipfile

with zipfile.ZipFile("data/ml-latest-small.zip", "r") as zip_ref:
    zip_ref.extractall("data/")
