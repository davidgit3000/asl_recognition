#!/bin/bash
# Download Kaggle ASL datasets to separate folders

# Create separate directories for each dataset
mkdir -p data/kaggle_asl1
mkdir -p data/kaggle_asl2

echo "Downloading ASL Alphabet dataset to kaggle_asl1..."
cd data/kaggle_asl1
kaggle datasets download -d grassknoted/asl-alphabet
unzip -q asl-alphabet.zip
rm asl-alphabet.zip
cd ../..

echo "Downloading ASL Dataset to kaggle_asl2..."
cd data/kaggle_asl2
kaggle datasets download -d ayuraj/asl-dataset
unzip -q asl-dataset.zip
rm asl-dataset.zip
cd ../..

echo "Done! Datasets downloaded to:"
echo "  - data/kaggle_asl1/ (grassknoted/asl-alphabet)"
echo "  - data/kaggle_asl2/ (ayuraj/asl-dataset)"
