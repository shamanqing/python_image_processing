from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2

import imgproc

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas



class MainWindow(QWidget):
    def __init__(self, title):
        super(QWidget, self).__init__()
        self.setWindowTitle(title)
        self.setMinimumSize(400, 300)

        self.srcImage, self.cvtImage = None, None
        self.srcImageView = QLabel('Source Image')
        self.cvtImageView = QLabel('Result Image')

        self.srcImageHistFigure = plt.figure()
        self.srcImageHistCanvas = FigureCanvas(self.srcImageHistFigure)

        self.cvtImageHistFigure = plt.figure()
        self.cvtImageHistCanvas = FigureCanvas(self.cvtImageHistFigure)

        self.cvtFunFigure = plt.figure()
        self.cvtFunCanvas = FigureCanvas(self.cvtFunFigure)

        imageView = QGridLayout()
        imageView.addWidget(self.srcImageView, 0, 0)
        imageView.addWidget(self.cvtImageView, 0, 2)
        imageView.addWidget(self.srcImageHistCanvas, 1, 0)
        imageView.addWidget(self.cvtFunCanvas, 1, 1)
        imageView.addWidget(self.cvtImageHistCanvas, 1, 2)


        openBtn = QPushButton('Open', self)
        openBtn.clicked.connect(self.openImage)

        histBtn = QPushButton('Hist', self)
        histBtn.clicked.connect(self.cvtHist)

        saveBtn = QPushButton('Save', self)
        saveBtn.clicked.connect(self.saveCvtImage)

        quitBtn = QPushButton('Quit', self)
        quitBtn.clicked.connect(qApp.quit)

        btnView = QHBoxLayout()
        btnView.addWidget(openBtn)
        btnView.addWidget(histBtn)
        btnView.addWidget(saveBtn)
        btnView.addWidget(quitBtn)


        mainLayout = QVBoxLayout()
        mainLayout.addLayout(imageView)
        mainLayout.addStretch(1)
        mainLayout.addLayout(btnView)

        self.setLayout(mainLayout)

    @pyqtSlot(bool)
    def saveCvtImage(self):
        if self.cvtImage is not None:
            cv2.imwrite('result.jpg', self.cvtImage)

    @pyqtSlot(bool)
    def cvtHist(self):
        if self.srcImage is not None:
            self.cvtImage, cdf = imgproc.do_hist(self.srcImage)
            showImage = QImage(self.cvtImage.data, self.cvtImage.shape[1],
                               self.cvtImage.shape[0], QImage.Format_Grayscale8)
            self.cvtImageView.setPixmap(QPixmap.fromImage(showImage))
            axes = self.cvtImageHistFigure.add_subplot(111)
            axes.hold(False)
            axes.hist(self.cvtImage.ravel(), 255)

            axes = self.cvtFunFigure.add_subplot(111)
            axes.hold(False)
            axes.plot(range(len(cdf)), cdf)

    @pyqtSlot(bool)
    def openImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 
                                                  'Image Files(*.jpg *.jpeg *.png *.tif *.tiff)')
        if len(fileName):
            self.srcImage = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
            showImage = QImage(self.srcImage.data, self.srcImage.shape[1],
                               self.srcImage.shape[0], QImage.Format_Grayscale8)

            self.srcImageView.setPixmap(QPixmap.fromImage(showImage))
            axes = self.srcImageHistFigure.add_subplot(111)
            axes.hold(False)
            axes.hist(self.srcImage.ravel(), 255)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindow('Image Processing')
    mw.show()
    app.exit(app.exec_())