import argparse
import multiprocessing
import os
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--hostfile", type=str, default="scripts/hostfile.txt")
parser.add_argument("--master_port", type=str, help="Master port", default="12219")

args = parser.parse_args()
current_working_directory = os.getcwd()

# Load hostfile
with open(args.hostfile, "r") as f:
    hostfile = [line.strip() for line in f.readlines()]

master_addr = hostfile[0]
master_port = args.master_port
n_nodes = len(hostfile)

def run_pdsh_command(host, rank):
    if rank == 0:
        process = subprocess.run(["pkill", "-f", "train*.py"])
    else:
        pdsh_command = f"pkill -f train*.py"
        process = subprocess.run(["ssh", host, pdsh_command])

    subprocesses.append(process)


subprocesses = []


def start_processes():
    processes = []
    for rank, host in enumerate(hostfile):
        p = multiprocessing.Process(target=run_pdsh_command, args=(host, rank))
        processes.append(p)
        p.start()
    return processes

try:
    processes = start_processes()
    for process in processes:
        process.join()
except KeyboardInterrupt:
    print("Terminating all processes...")
    for process in processes:
        process.kill()
    sys.exit(1)