from dataclasses import dataclass
import paramiko
from utils.console.colored import ColoredText
import time
import numpy as np
import inspect

@dataclass
class Remote(object):
    host: str
    username: str
    password: str
    port: int

    def __post_init__(self):
        self.client = None

    def __enter__(self):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(hostname=self.host, username=self.username, password=self.password, port=self.port)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client is not None:
            self.client.close()


host = '172.16.120.151'
username = 'ccdm'
password = 'cdm'
port = 999


if __name__ == "__main__":

    with Remote(host, username, password, port) as r:
        stdin, stdout, stderr = r.client.exec_command('ls /mnt/Local_data/Alexey_Zabolotniy/FingerMovementDecoder')
        data = stdout.read() + stderr.read()
        shell = r.client.invoke_shell()

        print(data)

    def b(a):
        return a+10
    def f(a: int = 2) -> int:
        text = ColoredText().style('r')('text')
        np.arange(100)
        time.time()
        return b(a**2)

    lines = inspect.getsource(f)
    print(lines)
    v = inspect.getclosurevars(f)


    for m in v.globals.values():
        # print(inspect.getmodule(m))
        # print(inspect.getsource(m))
        print(inspect.getmodule(m))
        # if inspect.getmodule(m).__name__ != '__main__':
        #     # print(inspect.getsource(m))
        #     if inspect.isclass(m):
        #         print(inspect.getmro(m))
        #         print(inspect.getmembers(m))
