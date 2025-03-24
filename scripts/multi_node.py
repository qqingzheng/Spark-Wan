import argparse
import multiprocessing
import os
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--hostfile", type=str, default="scripts/hostfile.txt")
parser.add_argument("--master_port", type=str, help="Master port", default="12219")
parser.add_argument("shell_file", type=str, help="Shell file to run")

args = parser.parse_args()
current_working_directory = os.getcwd()

relative_shell_file = args.shell_file
absolute_shell_file = os.path.join(current_working_directory, relative_shell_file)
if not os.path.exists(absolute_shell_file):
    raise FileNotFoundError(f"Shell file {absolute_shell_file} not found")

# Load hostfile
with open(args.hostfile, "r") as f:
    hostfile = [line.strip() for line in f.readlines()]

master_addr = hostfile[0]
master_port = args.master_port
n_nodes = len(hostfile)


def run_pdsh_command(host, rank):
    base_env = os.environ.copy()
    base_env["MASTER_ADDR"] = master_addr
    base_env["MASTER_PORT"] = master_port
    base_env["NNODES"] = str(n_nodes)
    base_env["NODE_RANK"] = str(rank)
    if rank == 0:
        process = subprocess.run(["bash", relative_shell_file], env=base_env)
    else:
        pdsh_command = (
            f"cd {current_working_directory} && "
            + " ".join([f"{key}='{value}'" for key, value in base_env.items()])
            + f" bash {relative_shell_file}"
        )
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