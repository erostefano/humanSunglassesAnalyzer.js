# Feature Engineering

[Kaggle Dataset: Sunglasses vs. No Sunglasses](https://www.kaggle.com/datasets/amol07/sunglasses-no-sunglasses)

| Dataset   | With Sunglasses | No Sunglasses | % More No Sunglasses |
|-----------|-----------------|---------------|----------------------|
| Train     | 1475            | 1776          | 20.5%                |
| Test      | 242             | 362           | 49.6%                |
| **Total** | **1717**        | **2138**      | **24.6%**            |

## Good

* All images have the same dimensions of 224x224 pixels.
* All images have the same color depth of 24-bit.

## Bad

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

| With Sunglasses | No Sunglasses |
|-----------------|---------------|
| 1000            | 1000          |
