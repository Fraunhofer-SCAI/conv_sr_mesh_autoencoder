import os

def mkdirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

