import os, os.path
from os import path
import matplotlib.pyplot as plt
import shutil
from PIL import Image
from resizeimage import resizeimage

def run():
    k = 18
    count = 0
    baseFol = "./all_navcam/labels_resize"
    saveFol = "./all_navcam/labels_resize_2"
    for root,dirs,files in os.walk("./all_navcam/labels_resize"):
        for f in files:
            im = Image.open(baseFol+"/"+f).convert('RGB')
            image = resizeimage.resize_contain(im, (514, 332))
            #plt.imshow(image)
            #plt.savefig(saveFol+"/"+f)
            image.save(saveFol+"/"+f)
    return




if __name__ == "__main__":
    run()
