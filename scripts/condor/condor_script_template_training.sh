#!/bin/sh

DIR=@dir@
TAG=@tag@
CFG=@cfg@
PYF=@pyf@

echo "Working directory: "$DIR
cd $DIR
echo "Moved to working directory."


python -u $PYF \
    --config "$CFG" \
    --tag "$TAG" \
