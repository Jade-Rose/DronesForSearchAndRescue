import socket
import threading
import time


class MultiTelloClass:
    def __init__(self):
        # IP and port of Tello
        self.tello1_address = ('192.168.0.45', 8889)
        self.tello2_address = ('192.168.0.46', 8889)

        # IP and port of local computer
        local1_address = ('', 9010)
        local2_address = ('', 9011)

        # Create a UDP connection that we'll send the command to
        self.sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Bind to the local address and port
        self.sock1.bind(local1_address)
        self.sock2.bind(local2_address)

        self.response1 = None
        self.response2 = None

        # listening thread for receiving cmd back
        receive_thread = threading.Thread(target=self.receiving_thread)
        receive_thread.daemon = True
        receive_thread.start()

    def sendCommand(self, command):
            encodedCommand = command.encode()
            self.sock1.sendto(encodedCommand, self.tello1_address)
            self.sock2.sendto(encodedCommand, self.tello2_address)

            currentTime = time.time()
            while time.time() - currentTime < 1:
                if self.response1 is not None:
                    response = self.response1.decode('utf-8')
                else:
                    response = 'none_response'

                self.response1 = None

                return response

    def receiving_thread(self):
            while True:
                try:
                    self.response1, ip = self.sock1.recvfrom(3000)
                    self.response2, ip_address = self.sock2.recvfrom(3000)
                    # print(self.response)
                except socket.error as exc:
                    print ("Caught exception socket.error : %s" % exc)

    def closeSockets(self):
            self.sendCommand('land')
            # Close the sockets
            self.sock1.close()
            self.sock2.close()
