#!/bin/bash

#get all files
list=($(ls ./documentation/diagrams))

#check images folder
if [ ! -d "./documentation/images" ]; then
    echo "Check folders: images folder does'nt exist. Creating it."
    mkdir ./documentation/images
else
    echo "Check folders: images folder exists."
fi

#check mmdc installation
if ! [ -x "$(command -v mmdc)" ]; then
    echo 'mmdc is not installed.' >&2
    exit 1
fi

#convert every file
for item in $list
do
filename="${item%.*}.png"
echo "Creating /documentation/images/$filename"
mmdc -i ./documentation/diagrams/$item -o ./documentation/images/$filename
done

#check pandoc installation
if ! [ -x "$(command -v pandoc)" ]; then
    echo 'pandoc is not installed.' >&2
    exit 1
fi

#create pdf if exists
pandoc -o ./documentation/documentation.pdf ./documentation/doc.md