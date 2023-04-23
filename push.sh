#!/bin/sh

git add .
if [[ -z $1 ]];
then 
    echo "No parameter passed. Default message loaded."
    git commit -am "made changes"
else
    git commit -am $1
fi
git push
echo "Pushed!"
