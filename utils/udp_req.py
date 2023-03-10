import socket
import select
import os
import tqdm
import time

UDP_IP = ''
buf = 1024

def udp_server(port, timeout = 5, start_flag=False):
    sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sk.bind((UDP_IP, port))
    if start_flag:
        rec, cli_addr = sk.recvfrom(buf)
    else:
        ready = select.select([sk], [], [], timeout)
        if ready[0]:
            rec, cli_addr = sk.recvfrom(buf)
        else:
            rec = b'complete'
            cli_addr = [None,None]
    return rec,cli_addr[0]

def udp_send(msg, ip, port):
    sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sk.sendto(msg,(ip,port))
