from MultiTelloClass import MultiTelloClass
import time
multiTello = MultiTelloClass()

# Initial search paramaters
searchLoops = 5 # how many times to loop around in search pattern
searchDistance = 20 # how far to travel along edge of expanding square
edgesTravelled = 0

multiTello.sendCommand('command')
multiTello.sendCommand('command')
multiTello.sendCommand('command')

time.sleep(2)
multiTello.sendCommand('takeoff')
print('told to takeoff')

while searchLoops > 0:
    print('working')
    # Send command then initiates a wait for 4 seconds to allow drone
    # to complete given command
    multiTello.sendCommand('forward ' + str(searchDistance))  # Travel along edge of search square
    currentTime = time.time()
    time.sleep(4)

    edgesTravelled += 1

    multiTello.sendCommand('cw 90')  # Turn corner of search square
    currentTime = time.time()
    time.sleep(4)

    # If drone has completed a loop then increase the distance it must travel
    # thus expanding the square
    if edgesTravelled >= 4:
        searchDistance = searchDistance * 2
        edgesTravelled = 0
        searchLoops -= 1

multiTello.sendCommand('land')

multiTello.closeSockets()