import os

test = "./all_navcam/output/test/left_jpgs/"
train = "./all_navcam/output/train/left_jpgs/"
val = "./all_navcam/output/val/left_jpgs/"

test = "./all_navcam/outputL/test/labels_pngs/"
train = "./all_navcam/outputL/train/labels_pngs/"
val = "./all_navcam/outputL/val/labels_pngs/"


totalFiles1 = 0
totalDir1 = 0

for base, dirs, files in os.walk(train):
    print("Searching in : ", base)
    for directories in dirs:
        totalDir1 += 1
    for Files in files:
        totalFiles1 += 1

totalFiles2 = 0
totalDir2 = 0

for base, dirs, files in os.walk(val):
    print("Searching in : ", base)
    for directories in dirs:
        totalDir2 += 1
    for Files in files:
        totalFiles2 += 1

totalFiles3 = 0
totalDir3 = 0

for base, dirs, files in os.walk(test):
    print("Searching in : ", base)
    for directories in dirs:
        totalDir3 += 1
    for Files in files:
        totalFiles3 += 1

print("Train")
print("Total number of files", totalFiles1)
print("Total Number of directories", totalDir1)
print("Total:", (totalDir1 + totalFiles1))

print("Val")
print("Total number of files", totalFiles2)
print("Total Number of directories", totalDir2)
print("Total:", (totalDir2 + totalFiles2))

print("Test")
print("Total number of files", totalFiles3)
print("Total Number of directories", totalDir3)
print("Total:", (totalDir3 + totalFiles3))
