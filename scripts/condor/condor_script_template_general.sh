#!/bin/sh

DIR=@dir@

echo "Working directory: "$DIR
cd $DIR
echo "Moved to working directory."

@cmd@
