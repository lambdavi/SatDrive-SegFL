#!/bin/python
from pymegatools import Megatools
import subprocess
import os
mega = Megatools()
print("Version:", mega.version)
url = "https://mega.nz/file/dONCFICZ#Wtbbyxr9N0Yeqf6sTP9mPKH_aDwuRZ0IbUxWPI_zyYY"
# Get a file name from url
print(mega.filename(url))

# Downloading a file from url
mega.download(url)

subprocess.call(["unzip", "data.zip"])

os.remove("data.zip")