# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test_image_list.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image
import ntpath
#from PyQt5.QtWidgets import QFileDialog


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1096, 663)
        self.window = MainWindow
        self.defaultWinW = 1096
        self.defaultWinH = 663
        self.defaultLabelImageW = 801
        self.defaultLabelImageH = 571
        self.defaultWidthBC = 281
        self.defaultHeightBC = 651
        self.QFileDialog = QtWidgets.QFileDialog()   
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(810, 5, 281, 651))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.layoutButtonContainer = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.layoutButtonContainer.setContentsMargins(0, 0, 0, 0)
        self.layoutButtonContainer.setObjectName("layoutButtonContainer")
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
        self.ptsTable.setColumnCount(6)
        self.ptsTable.setRowCount(5)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setItem(0, 3, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setItem(0, 4, item)
        item = QtWidgets.QTableWidgetItem()
        self.ptsTable.setItem(0, 5, item)
        self.ptsTable.horizontalHeader().setVisible(True)
        self.ptsTable.horizontalHeader().setCascadingSectionResizes(False)
        self.ptsTable.horizontalHeader().setDefaultSectionSize(70)
        self.ptsTable.horizontalHeader().setMinimumSectionSize(30)
        self.ptsTable.horizontalHeader().setSortIndicatorShown(False)
        self.ptsTable.horizontalHeader().setStretchLastSection(True)
        self.ptsTable.verticalHeader().setVisible(False)
        self.ptsTable.verticalHeader().setCascadingSectionResizes(False)
        self.ptsTable.verticalHeader().setDefaultSectionSize(15)
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
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(5, 640, 801, 19))
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
        if files:
            print(self.filePath)
            print(fileNames)
            self.listImages.clear()
            self.listImages.addItems(fileNames)
    
    def getPos(self , event):
        x = event.pos().x()
        y = event.pos().y() 
        #print(x, y)
        self.labelCoordinate.setText(str(int(round(x*self.widthRatio, 0)))+", "+str(int(round(y*self.heightRatio, 0))))
        
    
    def show_selected_item(self):
        item = self.listImages.currentItem()
        absFilePath = self.filePath+"/"+item.text()
        im = Image.open(absFilePath)
        self.widthRealIm, self.heightRealIm = im.size
        heightL = self.labelImage.height()
        widthL = self.labelImage.width()
        self.widthRatio = self.widthRealIm/widthL
        self.heightRatio = self.heightRealIm/heightL

        self.labelImage.setPixmap(QtGui.QPixmap(absFilePath))
        self.labelImage.mousePressEvent = self.getPos
        self.labelImage.mouseMoveEvent = self.getPos
        print(item.text())
    
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
        

    def someFunction(self):
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


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Annotator (Version 1.1)"))
        self.btnLoad.setText(_translate("MainWindow", "Load"))
        self.btnEntry.setText(_translate("MainWindow", "Entry"))
        self.btnCancel.setText(_translate("MainWindow", "Cancel"))
        self.btnSkip.setText(_translate("MainWindow", "Skip"))
        self.btnSave.setText(_translate("MainWindow", "Save"))
        item = self.ptsTable.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "1"))
        item = self.ptsTable.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "2"))
        item = self.ptsTable.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "3"))
        item = self.ptsTable.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "4"))
        item = self.ptsTable.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "5"))
        item = self.ptsTable.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Ob1"))
        item = self.ptsTable.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "P1"))
        item = self.ptsTable.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "P2"))
        item = self.ptsTable.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "P3"))
        item = self.ptsTable.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "P4"))
        item = self.ptsTable.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "P5"))
        __sortingEnabled = self.ptsTable.isSortingEnabled()
        self.ptsTable.setSortingEnabled(False)
        item = self.ptsTable.item(0, 0)
        item.setText(_translate("MainWindow", "Entry1"))
        item = self.ptsTable.item(0, 1)
        item.setText(_translate("MainWindow", "1024, 1221"))
        item = self.ptsTable.item(0, 2)
        item.setText(_translate("MainWindow", "2245, 1221"))
        item = self.ptsTable.item(0, 3)
        item.setText(_translate("MainWindow", "1234, 3113"))
        item = self.ptsTable.item(0, 4)
        item.setText(_translate("MainWindow", "1000, 2000"))
        item = self.ptsTable.item(0, 5)
        item.setText(_translate("MainWindow", "1718, 5839"))
        self.ptsTable.setSortingEnabled(__sortingEnabled)
        self.btnDeleteEntry.setText(_translate("MainWindow", "Delete Entry"))
        self.btnPrevious.setText(_translate("MainWindow", "Previous"))
        self.btnNext.setText(_translate("MainWindow", "Next"))
        self.labelCoordinate.setText(_translate("MainWindow", "X, Y"))
        
        self.btnLoad.clicked.connect(self.openFileNamesDialog)
        #self.listImages.clicked.connect(self.show_selected_item)
        self.listImages.itemSelectionChanged.connect(self.show_selected_item)
        self.btnNext.clicked.connect(self.next_image)
        self.btnPrevious.clicked.connect(self.prev_image)
        MainWindow.resized.connect(self.someFunction)



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


        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    #MainWindow = QtWidgets.QMainWindow()
    w = Window()
    ui = Ui_MainWindow()
    ui.setupUi(w)
    #MainWindow.show()

    w.show()
    sys.exit(app.exec_())
