#
# Code based on sample code provided by Ryze via Official Github repository under MIT License
# Copyright (c) 2018 Ryze
#
import tello
from tello_control_ui import TelloUI
import cv2


def main():

    drone = tello.Tello('', 8889)  
    vplayer = TelloUI(drone,"./img/")

	# start the Tkinter mainloop
    vplayer.root.mainloop()


if __name__ == "__main__":
    main()
