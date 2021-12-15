# Checks for GPU presence in the current machine

import torch

def run():
    print("Testing")
    print(torch.cuda.is_available())
    print("Done")
    return

run()
