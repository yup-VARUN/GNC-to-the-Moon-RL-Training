from torch.multiprocessing import Process
import os
import time

def dummy_process():
    print("Child printing own pid: ",os.getpid())
    time.sleep(10)
    return

def main():
    child_process = Process(target=dummy_process)

    # child_process.daemon = True
    child_process.start()
    print("Parent printing child's pid",child_process.ident)
    print("Waiting for child:")
    child_process.join()
    print("Wait Done")

if __name__ == '__main__':
    main()