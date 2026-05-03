## 1. Setting up a virtual environment

    $ git clone https://github.com/Vashisht-Patel/ST_1_Group_Project.git && cd ST_1_Group_Project/Macro

### a. Python environment

    $ python3 -m venv .st1env
    $ source .st1env/bin/activate

### a. Conda environment
    
    $ conda create -n st1 python=3.9
    $ conda activate st1

### b. Install required packages

    $ pip install -r Requirements.txt

## 2. Import classification dataset

Move dataset into the data folder.
Expected data format:
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
    
## 3. Run the script

From the Macro folder, run:

    $ python SRC/main.py
