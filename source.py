import socket
import time

ADDRESS = "0.0.0.0"
PORT = 8080


tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (ADDRESS, PORT)
tcp_socket.bind(server_address)
tcp_socket.listen(1)

while True:
    print("Waiting for connection")
    connection, client = tcp_socket.accept()

    try:
        print("Connected to client IP: {}".format(client))

        counter = 0
        while True:
            counter+=1
            data = f"{counter}, Hello, World\n"
            print(f"Sending data {data}")
            connection.sendall(str.encode(data))
            time.sleep(.5)

    finally:
        connection.close()