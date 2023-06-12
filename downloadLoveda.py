#!/bin/python
from pymegatools import Megatools
import subprocess
import os
mega = Megatools()
print("Version:", mega.version)
url = "https://mega.nz/file/QSUUDSyA#uYk3VksViA9vb0dQhSSA0fYAMdG6PL84mT5ShOnfipQ"
# Get a file name from url
print(mega.filename(url))

# Downloading a file from url
mega.download(url)

subprocess.call(["unzip", "data.zip"])

os.remove("data.zip")