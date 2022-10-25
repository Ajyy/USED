#!/bin/bash
# convert wv1 files to wav format in another folder.
echo "Converting wv1 files to wav format"
if $# -le 2
then
  echo
  echo Usage: conv.sh source_folder dest_folder
  echo
else
	#mkdir $2
	cd $2
	for f in $1/*/*.wv1
	do
	  filename="${f%%.*}"
	  filename=$(basename $filename)
	  sox -t wav "$f" "${filename}.wav"
	done
fi