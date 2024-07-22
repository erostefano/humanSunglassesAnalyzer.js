# Feature Engineering

[Kaggle Dataset: Sunglasses vs. No Sunglasses](https://www.kaggle.com/datasets/amol07/sunglasses-no-sunglasses)

| Dataset   | With Glasses | No Glasses | % More No Glasses |
|-----------|--------------|------------|-------------------|
| Train     | 1475         | 1776       | 20.5%             |
| Test      | 242          | 362        | 49.6%             |
| **Total** | **1717**     | **2138**   | **24.6%**         |

## Positive Aspects

* All images have the same dimensions of 224x224 pixels.
* Each image uses a 24-bit color depth.

## Negative Aspects (Before Fixes)

* The dataset is unbalanced, with 24.6% more images without glasses.
* The train-test split is 80:20 instead of the more standard 66:33.
* It is unclear if the dataset is balanced regarding gender, age and ethnicity.
* Kaggle shows warnings about broken images.
* The dataset contains multiple data types.
* File naming lacks a structured format.

## Fixes Implemented

* Balanced the dataset features.
* Removed the previous train-test split.
* Kept only one data type.
* Renamed the files for consistency.
