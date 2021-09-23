import os, os.path
from os import path
import matplotlib.pyplot as plt
import shutil

def run():
    k = 18
    count = 0
    baseFol = "./all_navcam/labels_pngs"
    for root,dirs,files in os.walk("./all_navcam/outputC/clusters"+str(k)+"/test"):
        for f in files:
            if root[44] != "L":
                name = f[:-4]
                folder = root[:44]+"L"+root[44:]
            #print(baseFol+"/"+name+".png")
            #plt.imshow(baseFol+"/"+name+".png")
            #plt.savefig(folder+"/"+name+".png")
                original = baseFol+"/"+name+".png"
                target = folder+"/"+name+".png"
                shutil.copyfile(original,target)
            #print(root)
            #print(f)
        #for f in files:
        #    print(f)
            #adj = f[:-11]+'.jpg'
            #newPath = "./all_navcam/left_resize/"+adj
            #if os.path.isfile(newPath):
            #    print("Yay!")
            #else:
            #    os.remove(newPath)
            #    count += 1
    #print(count)
    return




if __name__ == "__main__":
    run()
