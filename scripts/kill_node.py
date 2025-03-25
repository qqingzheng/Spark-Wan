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
        subprocess.run(["pkill", "-f", "train_step_distill.py"])
        subprocess.run(["pkill", "-f", "ConsisID"])
    else:
        subprocess.run(["ssh", host, "pkill", "-f", "train_step_distill.py"])
        subprocess.run(["ssh", host, "pkill", "-f", "ConsisID"])

for i, host in enumerate(hostfile):
    run_pdsh_command(host, i)
