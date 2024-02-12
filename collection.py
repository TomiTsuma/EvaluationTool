import argparse
from paramiko import SSHClient
from scp import SCPClient
from datetime import datetime
import os


def scpConnection(ip, chemical):
    # _{str(datetime.now()).replace('-','').replace(' ','_').replace(':','')
    path = f"{chemical}"
    print(path)
    try:
        os.mkdir(path)
    except Exception as e:
        print(e)
    try:
        os.chdir(os.path.join(os.getcwd(), path))
        ssh = SSHClient()
        ssh.load_system_host_keys()
        print(ip)
        ssh.connect(ip, username="root")
        scp = SCPClient(ssh.get_transport())
        scp.get(
            f"/home/tom/data_input/MSSC_DVC/output/saved_model/saved_models", recursive=True)
        scp.close()
    except Exception as e:
        print(e)
