# env-AIST

## Introduction
The goal is to have a GCN model that is able to identify the type of grasp the human uses to grab some common objects and to transfer this information to a shadow hand based on joint coordinates and their corresponding graphs. 

## Installation of work environment
Install environment tool for python version on system. You wil have to be inside this environment everytime you want to run the code
```bash

sudo apt install python3.10-venv

#create new env, in this exampple "pygenie2" is the name of the environment but choose whatever name fits best for you.

python3.10 -m venv pygenie2

# to activate the environment

source pygenie2/bin/activate

```


## Installation of libraries

Install other needed tools for the project.

NOTE: If some suggestion on updating pip comes up, ignore it as it may crash (up to december 2023 it was a problem). Unless it's strictly necessary the recommendation is to avoid it.

```bash
pip install numpy
pip install pandas
pip install networkx
pip3 install torch torchvision torchaudio
pip install torch_geometric
```

## Install project
If you are a collaborator on the project you will be able to see it and clone it, to do so through HTTPS simply do

```bash
git clone https://github.com/nadialeal/env-AIST.git
```

## To run the code
One you have cloned the repo you can execute the main program 

```bash
python3 trainning.py
```
