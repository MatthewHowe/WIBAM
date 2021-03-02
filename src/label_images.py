import numpy
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

print("succ")

def window(xpos, ypos, width, height):
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(xpos, ypos, width, height)
    win.setWindowTitle("Labeller")

    label = QtWidgets.QLabel(win)
    label.setText("Label")
    label.move(50,50)


    win.show()
    sys.exit(app.exec_())

# if __name__ == "__main__":

window(200,200, 300, 300)