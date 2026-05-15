<!--
*******************************
Author: u3298551
Group: 3
Assessment: 3
Date: 04/05/2026
Programming: Created README
*******************************
-->

## 1. Setting up a virtual environment

```bash
git clone https://github.com/Vashisht-Patel/ST_1_Group_Project.git && cd ST_1_Group_Project
```

### a. Python environment

```bash
python3 -m venv .st1env
source .st1env/bin/activate
```
    
### b. Conda environment

```bash
conda create -n st1 python=3.9
conda activate st1
```
    
### Install required packages

```bash
pip install -r Requirements.txt
```
## 2. Import classification dataset

Move dataset into the data folder.
Expected data format (In this case, the dataset_name will be "Raw"):
```text
    {Dataset_name}/
    |
    |-- {class_name1}/
    |   |-- {image1}.png
    |   |-- {image2}.png
    |   |-- ...
    |   |
    |-- {class_name2}/
    |   |-- {image1}.png
    |   |-- {image2}.png
    |   |-- ...
```
## 3. Run the script

From the ST_1_Group_Project folder, run:
```bash
python SRC/main.py
```
