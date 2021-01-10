# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 12:52:53 2019

@author: Pronaya Prosun Das
"""
# Version 1.4: shortcut key added

from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image
import ntpath
from threading import Thread
import time
import pandas as pd
#from PyQt5.QtWidgets import QFileDialog


class Ui_AnnotationWindow(object):

    def __init__(self, mainWindow=None):
        self.mainWindow = mainWindow
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1115, 675)
        self.dfCoordsStorage = pd.DataFrame(columns=['baseFolderName', 'fileName', 'p1', 'p2', 'p3', 'p4'])
        self.window = MainWindow
        self.defaultWinW = self.window.frameGeometry().width()
        self.defaultWinH = self.window.frameGeometry().height()
        self.defaultLabelImageW = 801
        self.defaultLabelImageH = 571
        self.defaultWidthBC = 281
        self.defaultHeightBC = 651
        self.QFileDialog = QtWidgets.QFileDialog()   
        self.widthRealIm=0
        self.heightRealIm=0
        self.entryStart = False
        self._image = []

        self.entrySc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+E'), self.window)
        self.entrySc.activated.connect(self.getCoords)

        self.cancelSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+A'), self.window)
        self.cancelSc.activated.connect(self.cancelEntry)

        self.skipSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+K'), self.window)
        self.skipSc.activated.connect(self.skipCoords)

        self.saveSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+S'), self.window)
        self.saveSc.activated.connect(self.saveEntries)

        self.nextSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+N'), self.window)
        self.nextSc.activated.connect(self.next_image)

        self.prevSc = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+P'), self.window)
        self.prevSc.activated.connect(self.prev_image)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(810, 5, 281, 651))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.layoutButtonContainer = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.layoutButtonContainer.setContentsMargins(0, 0, 0, 0)
        self.layoutButtonContainer.setObjectName("layoutButtonContainer")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.btnBack = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnBack.setObjectName("btnBack")
        self.horizontalLayout.addWidget(self.btnBack)
        self.layoutButtonContainer.addLayout(self.horizontalLayout)
        self.listImages = QtWidgets.QListWidget(self.verticalLayoutWidget)
        self.listImages.setObjectName("listImages")
        self.layoutButtonContainer.addWidget(self.listImages)
        self.btnLoad = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnLoad.setObjectName("btnLoad")
        self.layoutButtonContainer.addWidget(self.btnLoad)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(-1, 10, -1, 50)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.btnEntry = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnEntry.setObjectName("btnEntry")
        self.horizontalLayout_4.addWidget(self.btnEntry)
        self.btnCancel = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnCancel.setObjectName("btnCancel")
        self.horizontalLayout_4.addWidget(self.btnCancel)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.btnSkip = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnSkip.setObjectName("btnSkip")
        self.horizontalLayout_5.addWidget(self.btnSkip)
        self.btnSave = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnSave.setObjectName("btnSave")
        self.horizontalLayout_5.addWidget(self.btnSave)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setContentsMargins(-1, 10, -1, -1)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.ptsTable = QtWidgets.QTableWidget(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.ptsTable.setFont(font)
        self.ptsTable.setObjectName("ptsTable")
        self.ptsTable.setColumnCount(4)
        self.ptsTable.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setHorizontalHeaderItem(3, item)
        self.ptsTable.horizontalHeader().setVisible(True)
        self.ptsTable.horizontalHeader().setCascadingSectionResizes(False)
        self.ptsTable.horizontalHeader().setDefaultSectionSize(70)
        self.ptsTable.horizontalHeader().setMinimumSectionSize(30)
        self.ptsTable.horizontalHeader().setSortIndicatorShown(False)
        self.ptsTable.horizontalHeader().setStretchLastSection(True)
        self.ptsTable.verticalHeader().setVisible(False)
        self.ptsTable.verticalHeader().setCascadingSectionResizes(False)
        self.ptsTable.verticalHeader().setDefaultSectionSize(23)
        self.ptsTable.verticalHeader().setSortIndicatorShown(False)
        self.ptsTable.verticalHeader().setStretchLastSection(False)
        self.verticalLayout_3.addWidget(self.ptsTable)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.btnDeleteEntry = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnDeleteEntry.setObjectName("btnDeleteEntry")
        self.horizontalLayout_6.addWidget(self.btnDeleteEntry)
        self.verticalLayout_3.addLayout(self.horizontalLayout_6)
        self.verticalLayout_2.addLayout(self.verticalLayout_3)
        self.layoutButtonContainer.addLayout(self.verticalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.btnPrevious = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnPrevious.setObjectName("btnPrevious")
        self.horizontalLayout_3.addWidget(self.btnPrevious)
        self.btnNext = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.btnNext.setObjectName("btnNext")
        self.horizontalLayout_3.addWidget(self.btnNext)
        self.layoutButtonContainer.addLayout(self.horizontalLayout_3)
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(3, 5, 801, 571))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.layoutImageContainer = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.layoutImageContainer.setContentsMargins(0, 0, 4, 0)
        self.layoutImageContainer.setObjectName("layoutImageContainer")
        self.labelImage = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.labelImage.sizePolicy().hasHeightForWidth())
        self.labelImage.setSizePolicy(sizePolicy)
        self.labelImage.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.labelImage.setFrameShape(QtWidgets.QFrame.Box)
        self.labelImage.setText("")
        #self.labelImage.setPixmap(QtGui.QPixmap("cat.jpg"))
        self.labelImage.setScaledContents(True)
        self.labelImage.setObjectName("labelImage")
        self.layoutImageContainer.addWidget(self.labelImage)
        self.horizontalLayoutWidget_5 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(0, 640, 801, 19))
        self.horizontalLayoutWidget_5.setObjectName("horizontalLayoutWidget_5")
        self.layoutInfoContainer = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_5)
        self.layoutInfoContainer.setContentsMargins(0, 0, 0, 0)
        self.layoutInfoContainer.setObjectName("layoutInfoContainer")
        self.labelCoordinate = QtWidgets.QLabel(self.horizontalLayoutWidget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelCoordinate.sizePolicy().hasHeightForWidth())
        self.labelCoordinate.setSizePolicy(sizePolicy)
        self.labelCoordinate.setBaseSize(QtCore.QSize(0, 0))
        self.labelCoordinate.setObjectName("labelCoordinate")
        self.layoutInfoContainer.addWidget(self.labelCoordinate)
        self.labelInfo = QtWidgets.QLabel(self.horizontalLayoutWidget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(8)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelInfo.sizePolicy().hasHeightForWidth())
        self.labelInfo.setSizePolicy(sizePolicy)
        self.labelInfo.setText("")
        self.labelInfo.setObjectName("labelInfo")
        self.layoutInfoContainer.addWidget(self.labelInfo)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    def path_leaf(self, path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)
    
    def path_(self, path):
        head, tail = ntpath.split(path)
        return head

    def openFileNamesDialog(self):
        options = self.QFileDialog.Options()
        options |= self.QFileDialog.DontUseNativeDialog
        files, _ = self.QFileDialog.getOpenFileNames(self.window, "Open Files", 
                                                      "~/Work/Educational info/Gottingen/Thesis",
                                                      "Images JPG (*.jpg);;Images PNG (*.png);;All Files (*)") #,options=options)
        fileNames = [self.path_leaf(file_) for file_ in files]
        self.filePath = self.path_(files[0])
        self.baseFolderName = ntpath.basename(self.filePath)
        if files:
            print(self.filePath)
            print(self.baseFolderName)
            print(fileNames)
            self.listImages.clear()
            self.listImages.addItems(fileNames)
            try:
                self.dfCoordsStorage = pd.read_csv(self.filePath+"Annotation.csv")
                print(self.dfCoordsStorage.head())
            except:
                print(f"{bcolors.FAIL}Exception: Annotation loading failed!{bcolors.ENDC}")
                print(self.dfCoordsStorage.head())
            self.next_image()
    
    def getPos(self , event):
        x = event.pos().x()
        y = event.pos().y() 
        #print(x, y)
        self.labelCoordinate.setText(str(int(round(x*self.widthRatio, 0)))+", "+str(int(round(y*self.heightRatio, 0))))
        
        #Segment for Coordinate entry
        if(self.entryStart==True and len(self.xCoords)<=4):
            self.xCoords.append(int(round(x*self.widthRatio, 0)))
            self.yCoords.append(int(round(y*self.heightRatio, 0)))
            labelInfoStr = ""
            for i in range(0,len(self.xCoords)):
                labelInfoStr = labelInfoStr+"("+str(self.xCoords[i])+", "+str(self.yCoords[i])+")   "
            self.labelInfo.setText(labelInfoStr)
            
            ##if 4 points are taken
            if(len(self.xCoords)==4):
                self.btnLoad.setEnabled(True)
                self.btnNext.setEnabled(True)
                self.btnPrevious.setEnabled(True)
                self.btnEntry.setEnabled(True)
                self.listImages.setEnabled(True)
                
                coordsDict = {'baseFolderName': self.baseFolderName, 'fileName': self.listImages.currentItem().text(),
                              'p1': str(self.xCoords[0])+":"+str(self.yCoords[0]),
                              'p2': str(self.xCoords[1])+":"+str(self.yCoords[1]),
                              'p3': str(self.xCoords[2])+":"+str(self.yCoords[2]),
                              'p4': str(self.xCoords[3])+":"+str(self.yCoords[3])
                              }
                self.dfCoordsStorage = self.dfCoordsStorage.append(coordsDict, ignore_index=True)
                #self.dfCoordsStorage.to_csv(self.filePath)
                self.entryStart = False
                print(self.entryStart)
                self.showTable()
                
 
    def getCoords(self):
        print("HI")
        self.xCoords = []
        self.yCoords = []
        self.labelInfo.setText("")
        self.entryStart = True
        print(self.entryStart)
        
        self.btnLoad.setEnabled(False)
        self.btnNext.setEnabled(False)
        self.btnPrevious.setEnabled(False)
        self.btnEntry.setEnabled(False)
        self.listImages.setEnabled(False)

        
        #Thread(target=self.threadFunGetCoords, args=()).start()
    
    def skipCoords(self):
        if(self.entryStart==True):
            self.xCoords.append(-1)
            self.yCoords.append(-1)
            labelInfoStr = ""
            for i in range(0,len(self.xCoords)):
                labelInfoStr = labelInfoStr+"("+str(self.xCoords[i])+", "+str(self.yCoords[i])+")   "
                self.labelInfo.setText(labelInfoStr)
            ##if 4 points are taken
            if(len(self.xCoords)==4):
                self.btnLoad.setEnabled(True)
                self.btnNext.setEnabled(True)
                self.btnPrevious.setEnabled(True)
                self.btnEntry.setEnabled(True)
                self.listImages.setEnabled(True)
                
                coordsDict = {'baseFolderName': self.baseFolderName, 'fileName': self.listImages.currentItem().text(),
                              'p1': str(self.xCoords[0])+":"+str(self.yCoords[0]),
                              'p2': str(self.xCoords[1])+":"+str(self.yCoords[1]),
                              'p3': str(self.xCoords[2])+":"+str(self.yCoords[2]),
                              'p4': str(self.xCoords[3])+":"+str(self.yCoords[3])
                              }
                self.dfCoordsStorage = self.dfCoordsStorage.append(coordsDict, ignore_index=True)
                #self.dfCoordsStorage.to_csv(self.filePath)
                self.entryStart = False
                print(self.entryStart)
                self.showTable()
                
    
    def cancelEntry(self):
        if(self.entryStart==True):
            self.xCoords = []
            self.yCoords = []
            self.labelInfo.setText("")
            self.entryStart = False
        
            self.btnLoad.setEnabled(True)
            self.btnNext.setEnabled(True)
            self.btnPrevious.setEnabled(True)
            self.btnEntry.setEnabled(True)
            self.listImages.setEnabled(True)
            
    def saveEntries(self):
        self.dfCoordsStorage.to_csv(self.filePath+"Annotation.csv", sep=",", index=False)
        print(self.dfCoordsStorage.head())
        print("Unique Images: ", len(self.dfCoordsStorage.fileName.unique()))
        print("Entries saved!")
        
        
    def showTable(self):
        tmpCoordsSel =  self.dfCoordsStorage.loc[(self.dfCoordsStorage['baseFolderName'] == self.baseFolderName) & 
                                                 (self.dfCoordsStorage['fileName'] == self.listImages.currentItem().text())]
        
        self.fileNamesForDf = []
        self.indexForDf = []
        self.ptsTable.setRowCount(0)
        for index, row in tmpCoordsSel.iterrows():
            rowPosition = self.ptsTable.rowCount()
            self.ptsTable.insertRow(rowPosition)
            self.ptsTable.setItem(rowPosition , 0, QtWidgets.QTableWidgetItem(row['p1']))
            self.ptsTable.setItem(rowPosition , 1, QtWidgets.QTableWidgetItem(row['p2']))
            self.ptsTable.setItem(rowPosition , 2, QtWidgets.QTableWidgetItem(row['p3']))
            self.ptsTable.setItem(rowPosition , 3, QtWidgets.QTableWidgetItem(row['p4']))
            self.fileNamesForDf.append(row['fileName'])
            self.indexForDf.append(index)
        
        self.paintEvent()

            
    def delEntry(self):
        indexList = []                                                          
        for tableIndex in self.ptsTable.selectionModel().selectedRows():       
            index = QtCore.QPersistentModelIndex(tableIndex)         
            indexList.append(index)
        
        for index in indexList:
            print(self.ptsTable.item(index.row(), 0).text())
            dfMainCoordsInd = self.dfCoordsStorage[(self.dfCoordsStorage.baseFolderName==self.baseFolderName) &
                                                   (self.dfCoordsStorage.fileName==str(self.listImages.currentItem().text())) &
                                                   (self.dfCoordsStorage.p1==str(self.ptsTable.item(index.row(), 0).text())) &
                                                   (self.dfCoordsStorage.p2==str(self.ptsTable.item(index.row(), 1).text())) &
                                                   (self.dfCoordsStorage.p3==str(self.ptsTable.item(index.row(), 2).text())) &
                                                   (self.dfCoordsStorage.p4==str(self.ptsTable.item(index.row(), 3).text()))].index
            self.dfCoordsStorage = self.dfCoordsStorage.drop(dfMainCoordsInd, inplace=False)                        
            self.ptsTable.removeRow(index.row())  
        
        self._image = QtGui.QPixmap(self.absFilePath)
        self.showTable()
        
    def show_selected_item(self):
        item = self.listImages.currentItem()
        self.absFilePath = self.filePath+"/"+item.text()
        im = Image.open(self.absFilePath)
        self.widthRealIm, self.heightRealIm = im.size
        heightL = self.labelImage.height()
        widthL = self.labelImage.width()
        self.widthRatio = self.widthRealIm/widthL
        self.heightRatio = self.heightRealIm/heightL
        self._image = QtGui.QPixmap(self.absFilePath)
        

        print(item.text())
        self.showTable()

        self.labelImage.setPixmap(self._image)
        self.labelImage.mousePressEvent = self.getPos
        #self.labelImage.mouseMoveEvent = self.getPos


    def paintEvent(self):
        painter = QtGui.QPainter(self._image)  
        #painter.drawPixmap(self._image)
        #painter.setPen(QtGui.QPen(QtCore.Qt.blue, 5))
        #painter.drawEllipse(350, 350, 70, 70)
        painter.setPen(QtGui.QPen(QtCore.Qt.green,  2, QtCore.Qt.SolidLine))
        #painter.setBrush(QtGui.QBrush(QtCore.Qt.red, QtCore.Qt.SolidPattern))
        
        self.ecllipseWidthRatio = 6/634
        self.ecllipseHeightRatio = 6/423
        radW = self.widthRealIm*self.ecllipseWidthRatio
        radH = self.heightRealIm*self.ecllipseHeightRatio
        
        tableMaxRow = self.ptsTable.rowCount()
        for ind in range(0, tableMaxRow):
            for col in range(0,4):
                if(col==0):
                    painter.setBrush(QtGui.QBrush(QtCore.Qt.red, QtCore.Qt.SolidPattern))
                elif(col==1):
                    painter.setBrush(QtGui.QBrush(QtCore.Qt.yellow, QtCore.Qt.SolidPattern))
                elif(col==2):
                    painter.setBrush(QtGui.QBrush(QtCore.Qt.cyan, QtCore.Qt.SolidPattern))
                elif(col==3):
                    painter.setBrush(QtGui.QBrush(QtCore.Qt.blue, QtCore.Qt.SolidPattern))
                else:
                    painter.setBrush(QtGui.QBrush(QtCore.Qt.darkRed, QtCore.Qt.SolidPattern))
                x1, y1 = str(self.ptsTable.item(ind, col).text()).split(':')
                painter.drawEllipse(QtCore.QPointF(int(x1), int(y1)), radW, radH)
        #painter = QtGui.QPainter(self)  
        #painter.drawPixmap(self.rect(), self._image)
        #pen = QtGui.QPen()
        #pen.setWidth(5)
        #painter.setPen(pen)
        #painter.setPen(QtGui.QPen(QtCore.Qt.blue, 5))
        #painter.drawEllipse(350, 350, 70, 70)
        self.labelImage.setPixmap(self._image)
        self.labelImage.mousePressEvent = self.getPos

    
    def next_image(self):
        itemIndex = self.listImages.currentRow()
        if(itemIndex+1 < self.listImages.count()):
            nextItemIndex = itemIndex+1
        else:
            nextItemIndex = itemIndex
        self.listImages.setCurrentRow(nextItemIndex)
        #self.listImages.setFocus(nextItemIndex)
        #print(nextItemIndex)
        
    def prev_image(self):
        itemIndex = self.listImages.currentRow()
        if(itemIndex-1 >= 0):
            nextItemIndex = itemIndex-1
        else:
            nextItemIndex = itemIndex
        self.listImages.setCurrentRow(nextItemIndex)
        #self.listImages.setFocus(nextItemIndex)
        #print(nextItemIndex)
        

    def resizeOps(self):
        windowWidth = self.window.frameGeometry().width()
        windowHeight = self.window.frameGeometry().height()

        labelImageWidth = self.verticalLayoutWidget_4.width()
        labelImageHeight = self.verticalLayoutWidget_4.height()
        
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(3, 5, (self.defaultLabelImageW/self.defaultWinW)*windowWidth,
                                                             (self.defaultLabelImageH/(self.defaultWinH+30))*windowHeight))

        self.widthRatio = self.widthRealIm/self.labelImage.width()
        self.heightRatio = self.heightRealIm/self.labelImage.height()
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(5, windowHeight-55, 801, 19))
        
        widthBC = (self.defaultWidthBC/self.defaultWinW)*windowWidth
        if(widthBC>431):
            widthBC = 431
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(self.verticalLayoutWidget_4.width()+7, 5, widthBC,
                                                           (self.defaultHeightBC/(self.defaultWinH+30))*windowHeight))
        
        print("W", windowWidth, windowHeight)
        print("L",labelImageWidth, labelImageHeight)
        print((self.defaultLabelImageW/self.defaultWinW)*windowWidth, 
              (self.defaultLabelImageH/self.defaultWinH)*windowHeight) 


    def backToPreviousWindow(self):
        self.window.hide()
        self.mainWindow.show()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Annotator (Version 1.4)"))  #Version 1.4: shortcut key added
        self.btnBack.setText(_translate("MainWindow", "<-Back"))
        self.btnLoad.setText(_translate("MainWindow", "Load"))
        self.btnEntry.setText(_translate("MainWindow", "&Entry"))
        self.btnEntry.setToolTip('Ctrl+E to activate coordinates entry.') 
        self.btnCancel.setText(_translate("MainWindow", "C&ancel"))
        self.btnCancel.setToolTip('Ctrl+A to cancel coordinates entry.') 
        self.btnSkip.setText(_translate("MainWindow", "S&kip"))
        self.btnSkip.setToolTip('Ctrl+K to skip a coordinate.') 
        self.btnSave.setText(_translate("MainWindow", "&Save"))
        self.btnSave.setToolTip('Ctrl+S to save all the entries.') 
        item = self.ptsTable.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "P1"))
        item = self.ptsTable.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "P2"))
        item = self.ptsTable.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "P3"))
        item = self.ptsTable.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "P4"))
        __sortingEnabled = self.ptsTable.isSortingEnabled()
        self.ptsTable.setSortingEnabled(False)
        self.ptsTable.setSortingEnabled(__sortingEnabled)
        self.btnDeleteEntry.setText(_translate("MainWindow", "Delete Entry"))
        self.btnPrevious.setText(_translate("MainWindow", "&Previous"))
        self.btnPrevious.setToolTip('Ctrl+P to select previous frame.') 
        self.btnNext.setText(_translate("MainWindow", "&Next"))
        self.btnNext.setToolTip('Ctrl+N to select next frame.') 
        self.labelCoordinate.setText(_translate("MainWindow", "X, Y"))
        
        self.btnLoad.clicked.connect(self.openFileNamesDialog)
        #self.listImages.clicked.connect(self.show_selected_item)
        self.listImages.itemSelectionChanged.connect(self.show_selected_item)
        self.btnNext.clicked.connect(self.next_image)
        self.btnPrevious.clicked.connect(self.prev_image)
        self.btnEntry.clicked.connect(self.getCoords)
        self.btnSkip.clicked.connect(self.skipCoords)
        self.btnCancel.clicked.connect(self.cancelEntry)
        self.btnSave.clicked.connect(self.saveEntries)
        self.btnDeleteEntry.clicked.connect(self.delEntry)
        MainWindow.resized.connect(self.resizeOps)
        self.btnBack.clicked.connect(self.backToPreviousWindow)

        if(self.mainWindow==None):
            self.btnBack.setEnabled(False)



class Window(QtWidgets.QMainWindow):
    resized = QtCore.pyqtSignal()
    def  __init__(self, parent=None):
        super(Window, self).__init__(parent=parent)
        #ui = Ui_MainWindow()
        #ui.setupUi(self)
        #self.resized.connect(self.someFunction)
        

    def resizeEvent(self, event):
        self.resized.emit()
        return super(Window, self).resizeEvent(event)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    #MainWindow = QtWidgets.QMainWindow()
    w = Window()
    ui = Ui_AnnotationWindow()
    ui.setupUi(w)
    #MainWindow.show()

    w.show()
    sys.exit(app.exec_())
