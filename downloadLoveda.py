#!/bin/python
from pymegatools import Megatools
import subprocess
import os
mega = Megatools()
print("Version:", mega.version)
url = "https://mega.nz/file/IOsXnBBB#Pq7-sAIMu9v_5C0x3gaF2-vgYvs4AaseIhHFRRR5cP0"
# Get a file name from url
print(mega.filename(url))

# Downloading a file from url
mega.download(url)

subprocess.call(["unzip", "data.zip"])

os.remove("data.zip")