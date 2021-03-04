# Dataset

The Five Roundabouts Dataset used in this project can be downloaded from [Five Roundabouts](http://its.acfr.usyd.edu.au/datasets/five-roundabouts-dataset/)

# Environment

## Anaconda vs python evironment
There are two options for the environment: using Anaconda or making a simple python environment. The following sections will provide a tutorial for setting up either environment.

### Setting up Anaconda
We are going to use an Anaconda environment to build this project.

First, download miniconda from [miniconda](https://docs.conda.io/en/latest/miniconda.html). If you are using Windows, select Python 3.8 and a 64-bit installer.

### Setting up the environment
- Clone the project from github repository. 
- Run the environment.yml file with 
```
conda env create -f environment.yml
```
It creates a new Conda environment with the necessary libraries. 
- Make a new folder *src* into the environment. 
- Copy all files from the repository to src folder
- Activate the environment with 
``` 
conda activate thesis-dest
```
- Run the requirements.txt file in the environment to install the necessary libraries with pip. If you are using a GPU, run requirements-gpu.txt
```
pip install -r requirements.txt
```

## Python environment
Run the following command to create a virtual environment called "thesis-dest"

```
python3 -m venv thesis-dest
```

Activate the environment using

On macOS and Linux:
```
source thesis-dest/bin/activate
```

On Windows:
```
source .\thesis-dest\Scripts\activate
```

You can deactivate the environment using

```
deactivate
```
Install the required packages using
```
pip install -r requirements.txt
```



# Running the code
- Make sure you have the dataset file in the environment folder.
- Correct the path of the dataset in main.py is necessary
- Run main.py with 
```
python main.py
```
