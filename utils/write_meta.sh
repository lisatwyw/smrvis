#!/bin/bash

for d in {0..89}; do

echo $d

f=$(printf "%03d" "${d}" )
echo "
ObjectType = Image
NDims = 3
BinaryData = True
BinaryDataByteOrderMSB = False
CompressedData = False
TransformMatrix = 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0
Offset = 0.0 0.0 0.0
CenterOfRotation = 0.0 0.0 0.0
AnatomicalOrientation = RAI
ElementSpacing = 0.49479 0.49479 0.3125
DimSize = 768 768 1280
ElementType = MET_USHORT
ElementDataFile = scan_${f}.raw" > scan_${f}.mhd
done

