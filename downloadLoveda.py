#!/bin/python
from pymegatools import Megatools
import subprocess
import os
mega = Megatools()
print("Version:", mega.version)
url = "https://mega.nz/file/UeFAwAaC#wdJMiLsBUj4sfNWRSyNR35lgVvk-juBib1fu-3rGX0k"
# Get a file name from url
print(mega.filename(url))

# Downloading a file from url
mega.download(url)

subprocess.call(["unzip", "data.zip"])

os.remove("data.zip")