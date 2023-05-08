#!/bin/python
from pymegatools import Megatools
import subprocess
import os
mega = Megatools()
print("Version:", mega.version)
url = "https://mega.nz/file/ERkiQBaY#h-wktK7U7MpIG5nf-rMWF7d76NEM5ae_MrAmELftNR0"
# Get a file name from url
print(mega.filename(url))

# Downloading a file from url
mega.download(url)

subprocess.call(["unzip", "data.zip"])

os.remove("data.zip")
os.rename("data/GTA5", "data/gta5")