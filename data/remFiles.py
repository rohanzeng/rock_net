import os, os.path
from os import path

def run():
    count = 0
    for root,_,files in os.walk("./all_navcam/labels_resize"):
        for f in files:
            adj = f[:-11]+'.jpg'
            newPath = "./all_navcam/left_resize/"+adj
            if os.path.isfile(newPath):
                print("Yay!")
            else:
                os.remove(newPath)
                count += 1
    print(count)
    return




if __name__ == "__main__":
    run()
