import splitfolders

def run():
    splitfolders.ratio("./all_navcam/outputC/clusters18a", output = "./all_navcam/outputC/clusters18", seed=1337, ratio=(0.7, 0.15, 0.15))
    return

if __name__ == "__main__":
    run()

