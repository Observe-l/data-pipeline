import socket
import os
import tqdm
import time

UDP_IP = ''
UDP_PORT = 4700
buf = 1024

def udp_server():
    sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sk.bind((UDP_IP, UDP_PORT))
    rec, cli_addr = sk.recvfrom(buf)
    return rec,cli_addr[0]

def udp_send(msg, ip):
    sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sk.sendto(msg,(ip,UDP_PORT))
