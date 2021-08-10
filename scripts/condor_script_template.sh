#!/bin/sh

DIR=@dir@
TAG=@tag@
CFG=@cfg@

echo "Working directory: "$DIR
cd $DIR
echo "Moved to working directory."


python -u python/train.py \
    --config "$CFG" \
    --tag "$TAG" \
