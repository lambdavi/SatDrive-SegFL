#!/bin/bash
LINK_TO_GTA5="https://mega.nz/file/ERkiQBaY#h-wktK7U7MpIG5nf-rMWF7d76NEM5ae_MrAmELftNR0"
sudo apt-get update
sudo apt-get install megatools unzip -y
megadl $LINK_TO_GTA5
unzip -q *.zip
rm *.zip
