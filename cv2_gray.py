from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2



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

        grayBtn = QPushButton('Gray', self)
        grayBtn.clicked.connect(self.cvtGray)
        
        saveBtn = QPushButton('Save', self)
        saveBtn.clicked.connect(self.saveCvtImage)

        quitBtn = QPushButton('Quit', self)
        quitBtn.clicked.connect(qApp.quit)

        btnView = QHBoxLayout()
        btnView.addWidget(openBtn)
        btnView.addWidget(grayBtn)
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
    def cvtGray(self):
        if self.srcImage is not None:
            self.cvtImage = cv2.cvtColor(self.srcImage, cv2.COLOR_RGB2GRAY)

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
            self.srcImage = cv2.cvtColor(cv2.imread(fileName, 1), cv2.COLOR_BGR2RGB)
            showImage = QImage(self.srcImage.data, self.srcImage.shape[1],
                               self.srcImage.shape[0], QImage.Format_RGB888)

            self.srcImageView.setPixmap(QPixmap.fromImage(showImage))
            # self.srcImageView.setScaledContents(True)
            self.resize(showImage.width()*2, showImage.height())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindow('Image Processing')
    mw.show()
    app.exit(app.exec_())