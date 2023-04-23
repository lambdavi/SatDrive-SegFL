#!/bin/sh
git add .
if [[ -z $1 ]];
then 
    echo "No parameter passed. Default message loaded."
    git commit -m "made changes"
else
    git commit -m $1
fi
git push
echo "Press enter"
read
