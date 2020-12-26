#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 7 05:33:10 2020

@author: Pronaya Prosun Das
"""

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QProcess, QTextCodec, QDir
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QApplication, QPlainTextEdit
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
import torch.utils.data
import torch.nn as nn

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torchvision.utils

import pandas as pd
import numpy as np
import seaborn as sns
import os
import cv2
import threading as th
from contextlib import redirect_stdout

#from tracker import Tracker
import time
import imageio
import functools 
import pickle as pkl
import threading as th
import io  

# model_v4_preprocess_and_train contains model implementation. You should import your own script here.
import model_v4_preprocess_and_train as m4PT

# target for stdout - global variable used by sub-threads
out = io.StringIO()
terminationFlag = 0


class Ui_TrainWindow(object):
    def __init__(self, mainWindow=None):
        self.mainWindow = mainWindow
        
    def setupUi(self, TrainWindow):
        self.trainWindow = TrainWindow
        TrainWindow.setObjectName("TrainWindow")
        TrainWindow.resize(600, 685)
        TrainWindow.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        self.centralwidget = QtWidgets.QWidget(TrainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 10, 561, 661))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.btnBack = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnBack.setObjectName("btnBack")
        self.horizontalLayout_8.addWidget(self.btnBack)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_7 = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_4.addWidget(self.label_7)
        self.verticalLayout.addLayout(self.verticalLayout_4)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_5 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_2.addWidget(self.label_5)
        self.lineEditMatadataPath = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEditMatadataPath.setObjectName("lineEditMatadataPath")
        self.horizontalLayout_2.addWidget(self.lineEditMatadataPath)
        self.btnLoadMetadataPath = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnLoadMetadataPath.setObjectName("btnLoadMetadataPath")
        self.horizontalLayout_2.addWidget(self.btnLoadMetadataPath)
        self.verticalLayout_5.addLayout(self.horizontalLayout_2)
        self.verticalLayout.addLayout(self.verticalLayout_5)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_6 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout.addWidget(self.label_6)
        self.lineEditImageDirPath = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEditImageDirPath.setObjectName("lineEditImageDirPath")
        self.horizontalLayout.addWidget(self.lineEditImageDirPath)
        self.btnLoadImageDirPath = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnLoadImageDirPath.setObjectName("btnLoadImageDirPath")
        self.horizontalLayout.addWidget(self.btnLoadImageDirPath)
        self.verticalLayout_6.addLayout(self.horizontalLayout)
        self.verticalLayout.addLayout(self.verticalLayout_6)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.btnAddDataset = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnAddDataset.setObjectName("btnAddDataset")
        self.verticalLayout_3.addWidget(self.btnAddDataset)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_7.addWidget(self.label_3)
        self.listCSVPath = QtWidgets.QListWidget(self.verticalLayoutWidget)
        self.listCSVPath.setObjectName("listCSVPath")
        self.verticalLayout_7.addWidget(self.listCSVPath)
        self.horizontalLayout_3.addLayout(self.verticalLayout_7)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_8.addWidget(self.label_4)
        self.listImageDir = QtWidgets.QListWidget(self.verticalLayoutWidget)
        self.listImageDir.setObjectName("listImageDir")
        self.verticalLayout_8.addWidget(self.listImageDir)
        self.horizontalLayout_3.addLayout(self.verticalLayout_8)
        self.verticalLayout_9.addLayout(self.horizontalLayout_3)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.btnClearDataset = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnClearDataset.setObjectName("btnClearDataset")
        self.verticalLayout_10.addWidget(self.btnClearDataset)
        self.verticalLayout_9.addLayout(self.verticalLayout_10)
        self.verticalLayout.addLayout(self.verticalLayout_9)
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_11.addWidget(self.label)
        self.verticalLayout_12.addLayout(self.verticalLayout_11)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.radRetrain = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.radRetrain.setObjectName("radRetrain")
        self.horizontalLayout_4.addWidget(self.radRetrain)
        self.lineEditSavePath = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEditSavePath.setObjectName("lineEditSavePath")
        self.horizontalLayout_4.addWidget(self.lineEditSavePath)
        self.btnSaveModel = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnSaveModel.setObjectName("btnSaveModel")
        self.horizontalLayout_4.addWidget(self.btnSaveModel)
        self.verticalLayout_12.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.radAddTrain = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.radAddTrain.setObjectName("radAddTrain")
        self.horizontalLayout_5.addWidget(self.radAddTrain)
        self.lineEditLoadPath = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEditLoadPath.setObjectName("lineEditLoadPath")
        self.horizontalLayout_5.addWidget(self.lineEditLoadPath)
        self.btnLoadModel = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnLoadModel.setObjectName("btnLoadModel")
        self.horizontalLayout_5.addWidget(self.btnLoadModel)
        self.verticalLayout_12.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_13.addWidget(self.label_2)
        self.lineEditEpoch = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEditEpoch.setObjectName("lineEditEpoch")
        self.verticalLayout_13.addWidget(self.lineEditEpoch)
        self.horizontalLayout_6.addLayout(self.verticalLayout_13)
        self.verticalLayout_14 = QtWidgets.QVBoxLayout()
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.label_8 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_14.addWidget(self.label_8)
        self.lineEditTrainValSplit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEditTrainValSplit.setObjectName("lineEditTrainValSplit")
        self.verticalLayout_14.addWidget(self.lineEditTrainValSplit)
        self.horizontalLayout_6.addLayout(self.verticalLayout_14)
        self.verticalLayout_15 = QtWidgets.QVBoxLayout()
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.label_9 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_15.addWidget(self.label_9)
        self.lineEditBatch = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEditBatch.setObjectName("lineEditBatch")
        self.verticalLayout_15.addWidget(self.lineEditBatch)
        self.horizontalLayout_6.addLayout(self.verticalLayout_15)
        self.verticalLayout_12.addLayout(self.horizontalLayout_6)
        self.verticalLayout.addLayout(self.verticalLayout_12)
        self.verticalLayout_16 = QtWidgets.QVBoxLayout()
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem1)
        self.btnCancel = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnCancel.setObjectName("btnCancel")
        self.horizontalLayout_7.addWidget(self.btnCancel)
        self.btnStartTraining = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnStartTraining.setObjectName("btnStartTraining")
        self.horizontalLayout_7.addWidget(self.btnStartTraining)
        self.verticalLayout_16.addLayout(self.horizontalLayout_7)
        self.verticalLayout_17 = QtWidgets.QVBoxLayout()
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.textEditConsol = QtWidgets.QTextEdit(self.verticalLayoutWidget)
        self.textEditConsol.setObjectName("textEditConsol")
        self.verticalLayout_17.addWidget(self.textEditConsol)
        self.verticalLayout_16.addLayout(self.verticalLayout_17)
        self.verticalLayout.addLayout(self.verticalLayout_16)
        self.verticalLayout.setStretch(5, 10)
        self.verticalLayout.setStretch(7, 100)
        TrainWindow.setCentralWidget(self.centralwidget)
        
        self.user_defined_ini(TrainWindow) #call user defined changes 
        
        self.retranslateUi(TrainWindow)
        QtCore.QMetaObject.connectSlotsByName(TrainWindow)

    def retranslateUi(self, TrainWindow):
        _translate = QtCore.QCoreApplication.translate
        TrainWindow.setWindowTitle(_translate("TrainWindow", "TrainWindow"))        
        self.btnBack.setText(_translate("TrainWindow", "<-Back"))
        self.label_7.setText(_translate("TrainWindow", "Dataset:"))
        self.label_5.setText(_translate("TrainWindow", "Metadata Path"))
        self.btnLoadMetadataPath.setText(_translate("TrainWindow", "load"))
        self.label_6.setText(_translate("TrainWindow", "Image Directory Path"))
        self.btnLoadImageDirPath.setText(_translate("TrainWindow", "load"))
        self.btnAddDataset.setText(_translate("TrainWindow", "Add dataset"))
        self.label_3.setText(_translate("TrainWindow", "Metadata"))
        self.label_4.setText(_translate("TrainWindow", "Image Directories"))
        self.btnClearDataset.setText(_translate("TrainWindow", "Clear"))
        self.label.setText(_translate("TrainWindow", "Training Config:"))
        self.radRetrain.setText(_translate("TrainWindow", "Retrain"))
        self.btnSaveModel.setText(_translate("TrainWindow", "..."))
        self.radAddTrain.setText(_translate("TrainWindow", "Additive-train"))
        self.btnLoadModel.setText(_translate("TrainWindow", "load"))
        self.label_2.setText(_translate("TrainWindow", "Epoch"))
        self.lineEditEpoch.setText(_translate("TrainWindow", "10"))
        self.label_8.setText(_translate("TrainWindow", "Train/Val Split (%)"))
        self.lineEditTrainValSplit.setText(_translate("TrainWindow", "80"))
        self.label_9.setText(_translate("TrainWindow", "Batch Size"))
        self.lineEditBatch.setText(_translate("TrainWindow", "16"))
        self.btnCancel.setText(_translate("TrainWindow", "Cancel"))
        self.btnStartTraining.setText(_translate("TrainWindow", "Start "))
        
        self.user_defined_late_ini()


    def user_defined_ini(self, TrainWindow):
        self.isLoadModel = 0
        self.trainingStart = 0
        self.modelName = ""
        self.checkDatasetPath()
        self.checkModelParam()
        self.window = TrainWindow
        self.btnAddDataset.setEnabled(False)
        self.lineEditLoadPath.setEnabled(False)
        self.lineEditSavePath.setEnabled(False)
        self.btnSaveModel.setEnabled(False)
        self.btnLoadModel.setEnabled(False)
        self.btnCancel.setEnabled(False)
        #self.groupBox_2 = QtWidgets.QGroupBox("groupBox_2", self.centralwidget)
        #self.textEditConsol = QtWidgets.QTextEdit(self.centralwidget)
        #self.textEditConsol = MyConsole(self.centralwidget)
        #self.textEditConsol.setGeometry(QtCore.QRect(40, 506, 511, 121))
        #self.textEditConsol.setObjectName("textEditConsol")
        self.radRetrain.setChecked(True)
        # create a process output reader
        self.precessReader = ProcessOutputReader()

        # Thread to update text of output tab
        self.output_thread = OutputThread(self.centralwidget)
        self.output_thread.output_changed.connect(self.on_output_changed)
        # Start the listener
        self.output_thread.start()

        # Thread to fetch data
        self.fetch_data_thread = FetchDataThread()
        self.fetch_data_thread.finished.connect(self.thread_finished_action)
        if(self.mainWindow!=None):
            self.fetch_data_thread.finished.connect(lambda: self.btnBack.setEnabled(True))

    def thread_finished_action(self):
        global terminationFlag
        if(terminationFlag==1):
            self.on_output_changed("Cancelled!!!")
        self.btnCancel.setEnabled(False)
        self.btnStartTraining.setEnabled(True)
            
    def user_defined_late_ini(self):
        self.btnLoadMetadataPath.clicked.connect(self.loadMetadata)
        self.btnLoadImageDirPath.clicked.connect(self.loadImageDir)
        self.btnStartTraining.clicked.connect(self.showTrainingDetails)
        self.btnLoadModel.clicked.connect(self.loadModel)
        self.btnSaveModel.clicked.connect(self.saveModel)
        self.radRetrain.clicked.connect(self.checkRadioButtons)
        self.radAddTrain.clicked.connect(self.checkRadioButtons)
        self.checkRadioButtons()
        self.btnAddDataset.clicked.connect(self.addDataset)
        self.lineEditMatadataPath.textChanged.connect(self.checkDatasetPath)
        self.lineEditImageDirPath.textChanged.connect(self.checkDatasetPath)
        self.btnClearDataset.clicked.connect(self.clearDataset)
        self.lineEditEpoch.textChanged.connect(self.checkModelParam)
        self.lineEditTrainValSplit.textChanged.connect(self.checkModelParam)
        self.lineEditBatch.textChanged.connect(self.checkModelParam)
        self.btnCancel.clicked.connect(self.cancelTraining)
        self.btnBack.clicked.connect(self.backToPreviousWindow)


    def backToPreviousWindow(self):
        self.trainWindow.hide()
        self.mainWindow.show()

        
    #start button code
    def showTrainingDetails(self): 
        global terminationFlag
        self.trainingStart = self.trainingStart + 1
        
        #model name
        #dataset metadata csv
        #Epoch
        listCSVPathList =  [str(self.listCSVPath.item(i).text()) for i in range(self.listCSVPath.count())]
        listImageDirList =  [str(self.listImageDir.item(i).text()) for i in range(self.listImageDir.count())]
        sendStr = ""
        for x in zip(listCSVPathList, listImageDirList):
            sendStr = sendStr + x[0] + " > " + x[1] + " < "
             
        sendStr = sendStr + " | " + str(self.modelName) + " | " + str(self.lineEditEpoch.text()) + " | " + str(self.lineEditTrainValSplit.text()) + " | " + str(self.lineEditBatch.text()) + " | " + str(self.isLoadModel)
        print(sendStr)
        
        if not self.fetch_data_thread.isRunning():
            terminationFlag = 0
            # your back-end should have variable named terminationFlag. It will have value of 0 or 1. 
            # This is necessary for the cancel button to work. Based on the value of the terminationFlag,
            # you have to return/ terminate your model training.
            m4PT.terminationFlag = 0   
            self.btnBack.setEnabled(False)
            self.btnCancel.setEnabled(True)
            self.btnStartTraining.setEnabled(False)
            self.fetch_data_thread.update(sendStr)
            self.fetch_data_thread.start()
        #while 1:
        #    self.textEditConsol.setPlainText(strOut)
        #    time.sleep(1.5)

        #self.precessReader.produce_output.connect(self.textEditConsol.append_output)
        #self.precessReader.start('python3', ['../implementation (copy)/model_v4_data_preprocess.py', sendStr])  # start the process

    def on_output_changed(self, text):
        self.textEditConsol.append(text.strip())

        
    def cancelTraining(self):
        global terminationFlag
        terminationFlag = 1
        # your back-end should have variable named terminationFlag. It will have value of 0 or 1. 
        # This is necessary for the cancel button to work. Based on the value of the terminationFlag,
        # you have to return/ terminate your model training.
        m4PT.terminationFlag = 1
        self.on_output_changed("Cancelling the operation! Please wait...")
        self.btnCancel.setEnabled(False)
        
    def checkRadioButtons(self):
        if(self.radAddTrain.isChecked()==True):
            self.lineEditSavePath.setEnabled(False)
            self.lineEditLoadPath.setEnabled(True)
            self.btnSaveModel.setEnabled(False)
            self.btnLoadModel.setEnabled(True)
            self.lineEditSavePath.setText("")
        if(self.radRetrain.isChecked()==True):
            self.lineEditSavePath.setEnabled(True)
            self.lineEditLoadPath.setEnabled(False)
            self.btnSaveModel.setEnabled(True)
            self.btnLoadModel.setEnabled(False)
            self.lineEditLoadPath.setText("")
    
    def clearDataset(self):
        self.listCSVPath.clear()
        self.listImageDir.clear()
        
    def checkDatasetPath(self):
        if(len(self.lineEditMatadataPath.text())>1):
            self.metadataLoaded = True
        else:
            self.metadataLoaded = False
            
        if(len(self.lineEditImageDirPath.text())>1):
            self.imageDirLoaded = True
        else:
            self.imageDirLoaded = False
            
        if(self.metadataLoaded==True and self.imageDirLoaded==True):
            self.btnAddDataset.setEnabled(True)
        else:
            self.btnAddDataset.setEnabled(False)
            
    def addDataset(self):
        self.listCSVPath.addItem(self.lineEditMatadataPath.text())
        self.listImageDir.addItem(self.lineEditImageDirPath.text())
                
        self.btnAddDataset.setEnabled(False)
        self.lineEditMatadataPath.setText("")
        self.lineEditImageDirPath.setText("")
        
        
    def loadMetadata(self):
        self.currentMetadataPath, _ = QtWidgets.QFileDialog.getOpenFileName(self.window, "Load Metadata (CSV) ", "/home/",
                                                               "Model files (*.csv);;All Files (*)")
        if (self.currentMetadataPath):
            print(self.currentMetadataPath)
            self.currentMetadataPath = QDir.toNativeSeparators(self.currentMetadataPath)
            self.lineEditMatadataPath.setText(self.currentMetadataPath)
            
            self.checkDatasetPath()

    
    def loadImageDir(self):
        self.currentImageDirPath = QtWidgets.QFileDialog.getExistingDirectory(self.window, "Select Image Directory ", "/home/", 
                                                                                QtWidgets.QFileDialog.ShowDirsOnly)
        if (self.currentImageDirPath):
            print(self.currentImageDirPath)
            self.currentImageDirPath = QDir.toNativeSeparators(self.currentImageDirPath)
            self.lineEditImageDirPath.setText(self.currentImageDirPath)
            
            self.checkDatasetPath()
    
    def loadModel(self):
        self.modelName, _ = QtWidgets.QFileDialog.getOpenFileName(self.window, "Load Model ", "/home/",
                                                               "Model files (*.pyc);;All Files (*)")
        if (self.modelName):
            print(self.modelName)
            self.isLoadModel = 1
            self.modelName = QDir.toNativeSeparators(self.modelName)
            self.lineEditLoadPath.setText(self.modelName)
            self.checkModelParam()
    
    def saveModel(self):
        self.modelName, _ = QtWidgets.QFileDialog.getSaveFileName(self.window, "Save Model ", "/home/",
                                                               "Model files (*.pyc);;All Files (*)")
        if (self.modelName):
            print(self.modelName)
            self.isLoadModel = 0
            self.modelName = QDir.toNativeSeparators(self.modelName)
            self.lineEditSavePath.setText(self.modelName)
            self.checkModelParam()
            
    def checkModelParam(self):
        if(len(self.modelName)>1 and len(self.lineEditEpoch.text())>=1 and int(self.lineEditEpoch.text())>0 
           and len(self.lineEditTrainValSplit.text())>=1 and int(self.lineEditTrainValSplit.text())>0 
           and int(self.lineEditTrainValSplit.text())<=100 and len(self.lineEditBatch.text())>=1 
           and int(self.lineEditBatch.text())>0):
            self.btnStartTraining.setEnabled(True)
        else:
            self.btnStartTraining.setEnabled(False)
            

    

## Override  Classes

class ProcessOutputReader(QProcess):
    produce_output = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # merge stderr channel into stdout channel
        self.setProcessChannelMode(QProcess.MergedChannels)

        # prepare decoding process' output to Unicode
        codec = QTextCodec.codecForLocale()
        self._decoder_stdout = codec.makeDecoder()
        # only necessary when stderr channel isn't merged into stdout:
        # self._decoder_stderr = codec.makeDecoder()

        self.readyReadStandardOutput.connect(self._ready_read_standard_output)
        # only necessary when stderr channel isn't merged into stdout:
        # self.readyReadStandardError.connect(self._ready_read_standard_error)

    @pyqtSlot()
    def _ready_read_standard_output(self):
        raw_bytes = self.readAllStandardOutput()
        text = self._decoder_stdout.toUnicode(raw_bytes)
        self.produce_output.emit(text)

    # only necessary when stderr channel isn't merged into stdout:
    # @pyqtSlot()
    # def _ready_read_standard_error(self):
    #     raw_bytes = self.readAllStandardError()
    #     text = self._decoder_stderr.toUnicode(raw_bytes)
    #     self.produce_output.emit(text)


class MyConsole(QPlainTextEdit):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.setReadOnly(True)
        self.setMaximumBlockCount(10000)  # limit console to 10000 lines

        self._cursor_output = self.textCursor()

    @pyqtSlot(str)
    def append_output(self, text):
        self._cursor_output.insertText(str(text))
        self.scroll_to_last_line()

    def scroll_to_last_line(self):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.movePosition(QTextCursor.Up if cursor.atBlockStart() else
                            QTextCursor.StartOfLine)
        self.setTextCursor(cursor)



class FetchDataThread(QtCore.QThread):

    # Connection between this and the main thread.
    #data_fetched = QtCore.pyqtSignal(object, object)

    def __init__(self):
        super(FetchDataThread, self).__init__()

    def update(self, sendStr):
        self.sendStr = sendStr

    def run(self):

        # Can write directly to `out`
        #out.write('Fetching %s\n' % url)

        # This context manager will redirect the output from
        # `sys.stdout` to `out`
        with redirect_stdout(out):

            ### This is the place where you should call your training function which resides on the back-end script.
            ### self.sendStr contains all the given/selected parameter for the model. Therefore, you have  
            ### receive and parse it in your script. Use the parser function used in "model_v4_preprocess_and_train.py"
            m4PT.modelCall(self.sendStr)

        #out.write('='*80 + '\n')

        # Send data back to main thread
        #self.data_fetched.emit(url, data)


class OutputThread(QtCore.QThread):

    # This signal is sent when stdout captures a message
    # and triggers the update of the text box in the main thread.
    output_changed = QtCore.pyqtSignal(object)

    def run(self):
        global terminationFlag
        '''listener of changes to global `out`'''
        while True:
            if(terminationFlag==0):
                out.flush()
                text = out.getvalue()
                if text:
                    self.output_changed.emit(text)
                    # clear the buffer
                    out.truncate(0)
                    out.seek(0)
                time.sleep(1)

                





## Show the form
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    TrainWindow = QtWidgets.QMainWindow()
    ui = Ui_TrainWindow()
    ui.setupUi(TrainWindow)
    TrainWindow.show()
    sys.exit(app.exec_())
