#!/bin/bash
ORIGINALDIR=$(dirname $0)
DIR=$1
cd $DIR
mkdir ../wav_16khz
FILES=*
for f in $FILES
do
  echo "Processing ./$f file to ../wav_16khz/$f"
  # take action on each file. $f store current file name
  sox ./$f -r 16000 ../wav_16khz/$f
done
cd $ORIGINALDIR
