#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 04:36:05 2020

@author: Pronaya Prosun Das
"""


from PyQt5.QtCore import pyqtSignal, pyqtSlot, QProcess, QTextCodec, QDir
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import QApplication, QPlainTextEdit
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
from contextlib import redirect_stdout
import time
import io

# model_v4_analyse_and_track contains the function for tracking and counting interaction. You should import your own script here.
import model_v4_analyse_and_track as m4AT


out = io.StringIO()
terminationFlag = 0


class Ui_AnalyseWindow(object):
    def __init__(self, mainWindow=None):
        self.mainWindow = mainWindow
        
    def setupUi(self, AnalyseWindow):
        self.analyzeWindow = AnalyseWindow
        #self.analyzeWindow.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        AnalyseWindow.setObjectName("AnalyseWindow")
        AnalyseWindow.resize(565, 552)
        AnalyseWindow.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(AnalyseWindow.sizePolicy().hasHeightForWidth())
        AnalyseWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(AnalyseWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(18, 14, 521, 511))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.btnBack = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnBack.setObjectName("btnBack")
        self.horizontalLayout_4.addWidget(self.btnBack)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_5 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_6.addWidget(self.label_5)
        self.lineEditModelPath = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEditModelPath.setObjectName("lineEditModelPath")
        self.horizontalLayout_6.addWidget(self.lineEditModelPath)
        self.btnLoadModel = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnLoadModel.setObjectName("btnLoadModel")
        self.horizontalLayout_6.addWidget(self.btnLoadModel)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.lineEditVideoPath = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEditVideoPath.setObjectName("lineEditVideoPath")
        self.horizontalLayout.addWidget(self.lineEditVideoPath)
        self.btnLoadVideo = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnLoadVideo.setObjectName("btnLoadVideo")
        self.horizontalLayout.addWidget(self.btnLoadVideo)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_10 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_9.addWidget(self.label_10)
        self.lineEditOutputPath = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEditOutputPath.setObjectName("lineEditOutputPath")
        self.horizontalLayout_9.addWidget(self.lineEditOutputPath)
        self.btnSelectOutputPath = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnSelectOutputPath.setObjectName("btnSelectOutputPath")
        self.horizontalLayout_9.addWidget(self.btnSelectOutputPath)
        self.verticalLayout.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_6 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_7.addWidget(self.label_6)
        self.lineEditFrameToAnalyse = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEditFrameToAnalyse.setObjectName("lineEditFrameToAnalyse")
        self.horizontalLayout_7.addWidget(self.lineEditFrameToAnalyse)
        self.label_7 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_7.addWidget(self.label_7)
        self.lineEditTotalFrame = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEditTotalFrame.setObjectName("lineEditTotalFrame")
        self.horizontalLayout_7.addWidget(self.lineEditTotalFrame)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_2)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_9 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_8.addWidget(self.label_9)
        self.lineEditFrameRate = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEditFrameRate.setObjectName("lineEditFrameRate")
        self.horizontalLayout_8.addWidget(self.lineEditFrameRate)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem2)
        self.checkGenerateVideo = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkGenerateVideo.setCheckable(True)
        self.checkGenerateVideo.setChecked(False)
        self.checkGenerateVideo.setObjectName("checkGenerateVideo")
        self.horizontalLayout_8.addWidget(self.checkGenerateVideo)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem3)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.horizontalLayout_8.addLayout(self.horizontalLayout_3)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        spacerItem4 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem4)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        spacerItem5 = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem5)
        self.btnCancel = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnCancel.setObjectName("btnCancel")
        self.horizontalLayout_10.addWidget(self.btnCancel)
        self.btnStart = QtWidgets.QPushButton(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnStart.sizePolicy().hasHeightForWidth())
        self.btnStart.setSizePolicy(sizePolicy)
        self.btnStart.setObjectName("btnStart")
        self.horizontalLayout_10.addWidget(self.btnStart)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem6)
        self.verticalLayout.addLayout(self.horizontalLayout_10)
        spacerItem7 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.verticalLayout.addItem(spacerItem7)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.textEditConsol = QtWidgets.QTextEdit(self.verticalLayoutWidget)
        self.textEditConsol.setObjectName("textEditConsol")
        self.verticalLayout_2.addWidget(self.textEditConsol)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        AnalyseWindow.setCentralWidget(self.centralwidget)

        self.user_defined_ini(AnalyseWindow)
        
        self.retranslateUi(AnalyseWindow)
        QtCore.QMetaObject.connectSlotsByName(AnalyseWindow)

    def retranslateUi(self, AnalyseWindow):
        _translate = QtCore.QCoreApplication.translate
        AnalyseWindow.setWindowTitle(_translate("AnalyseWindow", "Analyse and Track"))           
        self.btnBack.setText(_translate("AnalyseWindow", "<- Back"))
        self.label_5.setText(_translate("AnalyseWindow", "Model: "))
        self.btnLoadModel.setText(_translate("AnalyseWindow", "load"))
        self.label.setText(_translate("AnalyseWindow", "Video: "))
        self.btnLoadVideo.setText(_translate("AnalyseWindow", "load"))
        self.label_10.setText(_translate("AnalyseWindow", "Output: "))
        self.btnSelectOutputPath.setText(_translate("AnalyseWindow", "Select"))
        self.label_6.setText(_translate("AnalyseWindow", "Frames to analyse: "))
        self.label_7.setText(_translate("AnalyseWindow", "to"))
        self.label_9.setText(_translate("AnalyseWindow", "Frame rate: "))
        self.checkGenerateVideo.setText(_translate("AnalyseWindow", "Generate Video"))
        self.btnCancel.setText(_translate("AnalyseWindow", "Cancel"))
        self.btnStart.setText(_translate("AnalyseWindow", "Start"))

        self.user_defined_late_ini()
    
    
    def user_defined_ini(self, AnalyseWindow):
        self.window = AnalyseWindow
        self.checkModelParam()
        self.btnCancel.setEnabled(False)
        if(self.mainWindow==None):
            self.btnBack.setEnabled(False)
        # Thread to update text of output tab
        self.output_thread = OutputThread(self.centralwidget)
        self.output_thread.output_changed.connect(self.on_output_changed)
        # Start the listener
        self.output_thread.start()

        # Thread to fetch data
        self.fetch_data_thread = FetchDataThread()
        self.fetch_data_thread.finished.connect(lambda: self.btnStart.setEnabled(True))
        self.fetch_data_thread.finished.connect(lambda: self.btnCancel.setEnabled(False))
        if(self.mainWindow!=None):
            self.fetch_data_thread.finished.connect(lambda: self.btnBack.setEnabled(True))


        
    def user_defined_late_ini(self):
        self.btnLoadModel.clicked.connect(self.loadModel)
        self.btnLoadVideo.clicked.connect(self.loadVideo)
        self.btnSelectOutputPath.clicked.connect(self.selectOutputDir)
        self.lineEditModelPath.textChanged.connect(self.checkModelParam)
        self.lineEditVideoPath.textChanged.connect(self.checkModelParam)
        self.lineEditOutputPath.textChanged.connect(self.checkModelParam)
        self.lineEditFrameToAnalyse.textChanged.connect(self.checkModelParam)
        self.lineEditTotalFrame.textChanged.connect(self.checkModelParam)
        self.lineEditFrameRate.textChanged.connect(self.checkModelParam)
        self.btnStart.clicked.connect(self.startAnalysis)
        self.btnCancel.clicked.connect(self.cancelAnalysis)
        self.btnBack.clicked.connect(self.backToPreviousWindow)


    def backToPreviousWindow(self):
        #from main import Ui_Dialog
        #mainDialog = QtWidgets.QDialog()
        #uiMain = Ui_Dialog()
        #uiMain.setupUi(mainDialog)
        #self.output_thread.quit()
        self.analyzeWindow.hide()
        self.mainWindow.show()
              
        
    def loadModel(self):
        self.modelName, _ = QtWidgets.QFileDialog.getOpenFileName(self.window, "Load Model ", "/home/",
                                                               "Model files (*.pyc);;All Files (*)")
        if (self.modelName):
            print(self.modelName)
            self.modelName = QDir.toNativeSeparators(self.modelName)
            self.lineEditModelPath.setText(self.modelName)
            self.checkModelParam()

    def loadVideo(self):
        self.videoName, _ = QtWidgets.QFileDialog.getOpenFileName(self.window, "Load Video ", "/home/",
                                                               "Video files - mp4 (*.mp4);;Video files - avi (*.avi);;All Files (*)")
        if (self.videoName):
            print(self.videoName)
            self.videoName = QDir.toNativeSeparators(self.videoName)
            self.lineEditVideoPath.setText(self.videoName)
            
            cap = cv2.VideoCapture(self.videoName)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Total frames: ", length)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print("FPS: ", fps)
            self.lineEditFrameToAnalyse.setText(str(0))
            self.lineEditTotalFrame.setText(str(length))
            self.lineEditFrameRate.setText(str(round(fps, 8)))
            self.checkModelParam()
    
    def selectOutputDir(self):
        self.outputDir = QtWidgets.QFileDialog.getExistingDirectory(self.window, "Select Output Directory ", "/home/", 
                                                                                QtWidgets.QFileDialog.ShowDirsOnly)
        if (self.outputDir):
            print(self.outputDir)
            self.outputDir = QDir.toNativeSeparators(self.outputDir)
            self.lineEditOutputPath.setText(self.outputDir)
            self.checkModelParam()
    
    def checkModelParam(self):
        if(len(self.lineEditModelPath.text())>=1 and len(self.lineEditVideoPath.text())>=1 
           and len(self.lineEditFrameToAnalyse.text())>=1 and int(self.lineEditFrameToAnalyse.text())>=0 
           and len(self.lineEditTotalFrame.text())>=1 and int(self.lineEditTotalFrame.text())>=0 
           and int(self.lineEditTotalFrame.text())>int(self.lineEditFrameToAnalyse.text()) 
           and len(self.lineEditFrameRate.text())>=1 and float(self.lineEditFrameRate.text())>0.0 
           and len(self.lineEditOutputPath.text())>=1):
            self.btnStart.setEnabled(True)
        else:
            self.btnStart.setEnabled(False)
            
    def startAnalysis(self):
        print("Start Analysis")
        checkGenVideo = 0
        if(self.checkGenerateVideo.isChecked()):
            checkGenVideo = 1
            
        sendStr = ""
        sendStr = sendStr + str(self.modelName) + " | " + str(self.lineEditVideoPath.text()) + " | " + str(self.lineEditOutputPath.text()) + " | " + str(self.lineEditFrameToAnalyse.text()) + " | " + str(self.lineEditTotalFrame.text()) + " | " + str(self.lineEditFrameRate.text()) + " | " + str(checkGenVideo)
        print(sendStr)
        terminationFlag = 0
        # your back-end should have variable named terminationFlag. It will have value of 0 or 1. 
        # This is necessary for the cancel button to work. Based on the value of the terminationFlag,
        # you have to return/ terminate your model training.
        m4AT.terminationFlag = 0

        if not self.fetch_data_thread.isRunning():
            self.btnCancel.setEnabled(True)
            self.btnStart.setEnabled(False)
            self.btnBack.setEnabled(False)
            self.output_thread.updateTerminationFlag_(0)
            self.fetch_data_thread.update(sendStr)
            self.fetch_data_thread.start()    
        
    def cancelAnalysis(self):
        print("Cancel analysis")
        terminationFlag = 1
        # your back-end should have variable named terminationFlag. It will have value of 0 or 1. 
        # This is necessary for the cancel button to work. Based on the value of the terminationFlag,
        # you have to return/ terminate your model training.
        m4AT.terminationFlag = 1
        #out.write("Cancelling the process. Please wait!")
        self.output_thread.updateTerminationFlag_(1)
        self.on_output_changed("Process cancelled. Please wait!")
        self.btnCancel.setEnabled(False)
        
    def on_output_changed(self, text):
        self.textEditConsol.append(text.strip())



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
            analyseVideo(self.sendStr)

        #out.write('='*80 + '\n')

        # Send data back to main thread
        #self.data_fetched.emit(url, data)


class OutputThread(QtCore.QThread):

    # This signal is sent when stdout captures a message
    # and triggers the update of the text box in the main thread.
    output_changed = QtCore.pyqtSignal(object)
    terminationFlag_ = 0
    def updateTerminationFlag_(self, terminationFlag_):
        self.terminationFlag_ = terminationFlag_
        
    def run(self):
        '''listener of changes to global `out`'''
        while True:
            if(self.terminationFlag_==0):
                out.flush()
                text = out.getvalue()
                if text:
                    self.output_changed.emit(text)
                    # clear the buffer
                    out.truncate(0)
                    out.seek(0)
                time.sleep(1)            


def analyseVideo(sendStr):
    print(sendStr)

    ### This is the place where you should call the tracker which resides on the back-end script.
    ### sendStr contains all the given/selected parameter for the back-end script. Therefore, you have  
    ### receive and parse it in your script. Use the parser function used in "model_v4_analyse_and_track.py"
    m4AT.main_func(sendStr)
    #del modelAE

            
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    AnalyseWindow = QtWidgets.QMainWindow()
    ui = Ui_AnalyseWindow()
    ui.setupUi(AnalyseWindow)
    AnalyseWindow.show()
    sys.exit(app.exec_())
