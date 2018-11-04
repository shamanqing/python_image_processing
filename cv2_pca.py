from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2

import imgproc



class MainWindow(QWidget):
    def __init__(self, title):
        super(QWidget, self).__init__()
        self.setWindowTitle(title)
        self.setMinimumSize(400, 300)

        self.srcImage, self.cvtImage = None, None
        self.srcImageView = QLabel('Source Image')
        self.cvtImageView = QLabel('Result Image')

        imageView = QHBoxLayout()
        imageView.addWidget(self.srcImageView)
        imageView.addStretch(1)
        imageView.addWidget(self.cvtImageView)

        openBtn = QPushButton('Open', self)
        openBtn.clicked.connect(self.openImage)

        saveBtn = QPushButton('Save', self)
        saveBtn.clicked.connect(self.saveCvtImage)

        quitBtn = QPushButton('Quit', self)
        quitBtn.clicked.connect(qApp.quit)

        btnView = QHBoxLayout()
        btnView.addWidget(openBtn)
        btnView.addWidget(saveBtn)
        btnView.addWidget(quitBtn)

        self.valueSilder = QSlider(Qt.Horizontal, self)
        self.valueSilder.setFocusPolicy(Qt.NoFocus)
        self.valueSilder.valueChanged[int].connect(self.cvtPca)

        btmView = QVBoxLayout()
        btmView.addLayout(btnView)
        btmView.addWidget(self.valueSilder)

        mainLayout = QVBoxLayout()
        mainLayout.addLayout(imageView)
        mainLayout.addStretch(1)
        mainLayout.addLayout(btmView)

        self.setLayout(mainLayout)

    @pyqtSlot(bool)
    def saveCvtImage(self):
        if self.cvtImage is not None:
            cv2.imwrite('result.jpg', self.cvtImage)

    @pyqtSlot(int)
    def cvtPca(self, value):
        if self.srcImage is not None:
            self.cvtImage = imgproc.do_pca(self.srcImage, value)
            showImage = QImage(self.cvtImage, self.cvtImage.shape[1],
                               self.cvtImage.shape[0], QImage.Format_Grayscale8)
            self.cvtImageView.setPixmap(QPixmap.fromImage(showImage))
            # self.cvtImageView.setScaledContents(True)
            self.resize(showImage.width()*2, showImage.height())

    @pyqtSlot(bool)
    def openImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 
                                                  'Image Files(*.jpg *.jpeg *.png *.tif *.tiff)')
        if len(fileName):
            self.srcImage = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
            showImage = QImage(self.srcImage.data, self.srcImage.shape[1],
                               self.srcImage.shape[0], QImage.Format_Grayscale8)

            self.srcImageView.setPixmap(QPixmap.fromImage(showImage))
            # self.srcImageView.setScaledContents(True)
            self.resize(showImage.width()*2, showImage.height())
            self.valueSilder.setMaximum(self.srcImage.shape[1])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindow('Image Processing')
    mw.show()
    app.exit(app.exec_())