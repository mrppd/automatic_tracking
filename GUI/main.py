#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 6 19:33:02 2020

@author: Pronaya Prosun Das
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from train_interface_ import Ui_TrainWindow
from analyse_ import Ui_AnalyseWindow
from image_annotator_v1_4 import Ui_AnnotationWindow, Window

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(454, 405)
        self.thisDialog = Dialog
        #self.thisDialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.verticalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(57, 40, 341, 321))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.btnAnnotation = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnAnnotation.setObjectName("btnAnnotation")
        self.verticalLayout.addWidget(self.btnAnnotation)
        self.btnTrain = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnTrain.setObjectName("btnTrain")
        self.verticalLayout.addWidget(self.btnTrain)
        self.btnDetectAndTrack = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnDetectAndTrack.setObjectName("btnDetectAndTrack")
        self.verticalLayout.addWidget(self.btnDetectAndTrack)
        
        self.userDefinedIni()
        
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.btnAnnotation.setText(_translate("Dialog", "Annotation"))
        self.btnTrain.setText(_translate("Dialog", "Train"))
        self.btnDetectAndTrack.setText(_translate("Dialog", "Detect and Track"))
        self.btnAnnotation.clicked.connect(self.openAnnotationWindow)
        self.btnTrain.clicked.connect(self.openTrainWindow)
        self.btnDetectAndTrack.clicked.connect(self.openAnalyseWindow)
    
    def userDefinedIni(self):
        self.analyseWindow = QtWidgets.QMainWindow()
        self.uiAnalyseWindow = Ui_AnalyseWindow(self.thisDialog)
        self.uiAnalyseWindow.setupUi(self.analyseWindow)
        
        self.trainWindow = QtWidgets.QMainWindow()
        self.uiTrainWindow = Ui_TrainWindow(self.thisDialog)
        self.uiTrainWindow.setupUi(self.trainWindow)
        
        self.annotationWindow = Window()
        self.uiAnnotationWindow = Ui_AnnotationWindow(self.thisDialog)
        self.uiAnnotationWindow.setupUi(self.annotationWindow)
        
        
    def openTrainWindow(self):
        self.thisDialog.hide()
        self.trainWindow.show()
    
    def openAnalyseWindow(self):
        self.thisDialog.hide()
        self.analyseWindow.show() 
    
    def openAnnotationWindow(self):
        self.thisDialog.hide()
        self.annotationWindow.show()         


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

