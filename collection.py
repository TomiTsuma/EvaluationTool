import argparse
from paramiko import SSHClient
from scp import SCPClient
from datetime import datetime
import os

ap = argparse.ArgumentParser()
ap.add_argument("-ip", "--ip-address", required=True, help="The IP address of the model droplet")
ap.add_argument("-c", "--chemical", required=True, help="The IP address of the model droplet")
args = vars(ap.parse_args())

def scpConnection():
    path = f"{args['chemical']}_{str(datetime.now()).replace('-','').replace(' ','_').replace(':','')}"
    print(path)
    try:
        os.mkdir(path)
    except Exception as e:
        print(e)
    os.chdir(os.path.join(os.getcwd(),path))
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(args['ip_address'],username="root", passphrase="Tobirama13")
    scp = SCPClient(ssh.get_transport())
    try:
        scp.get(f"/home/tom/data_input/MSSC_DVC/output", recursive=True)
    except Exception as e:
        print(e)
    scp.close()

print( args )
if __name__ == "__main__":
    scpConnection()