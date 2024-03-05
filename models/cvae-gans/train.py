import torch
import os
import argparse
import numpy as np
import time

def mkdir(path):
    if os.path.exists(path) == False :
        os.makedirs(path)

def mkdirs(paths):
    if type(paths) == str:  
        paths = [paths]      
    for path in paths:
        mkdir(path)







def main(arg):
    print("received arguments")
    print(arg)
    









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="description of the program")
    parser.add_argument('--param1',type = int,default  = 100,help = "description of parameter1")
    parser.add_argument('--param2',default = "yo",help = "description of parameter2")
    args = parser.parse_args()
    main(args)