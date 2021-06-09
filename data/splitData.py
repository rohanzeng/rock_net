import splitfolders

def run():
    splitfolders.ratio("./all_navcam/inputL", output = "./all_navcam/outputL", seed=1337, ratio=(0.7, 0.15, 0.15))
    return

if __name__ == "__main__":
    run()

