import sys
from PyQt5.QtWidgets import QApplication, QMessageBox,QWidget, QGridLayout, QSplitter, QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QFileDialog, QVBoxLayout, QHBoxLayout, QHeaderView, QTableWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import os
from PyQt5.QtGui import QImage
import numpy as np
# import TestOutput
from keras.models import load_model
import pickle
from collections import deque

class SportsVideoClassification(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1300,590)
        #self.reshape(binary_predictions, (-1, 1))
        self.item=''
        self.timer = QtCore.QTimer()
        self.summary_text=''
        self.output=[]
        self.q_img=''
        self.current_label=None
        self.label=''
        self.label_times={}
        self.timing=''
       
        self.setStyleSheet('background-color: #f2d8ee;')
        # UI elements for selecting and playing a video
        self.input_path_textfield = QLineEdit(self)
        self.input_path_textfield.setReadOnly(True)
        self.input_path_textfield.setGeometry(QtCore.QRect(10, 450, 400, 23))
        self.input_path_button = QPushButton('SELECT VIDEO', self)
        self.input_path_button.clicked.connect(self.select_video)
        self.input_path_button.setGeometry(QtCore.QRect(100, 490, 85, 23))

        self.reset_button = QPushButton('CLEAR', self)
        self.reset_button.clicked.connect(self.reset_ui)
        self.reset_button.setGeometry(QtCore.QRect(630, 490, 75, 23))
        self.reset_button.setEnabled(False)

        self.line = QtWidgets.QFrame(self)
        self.line.setGeometry(QtCore.QRect(420, 0, 31, 571))
        self.line.setSizeIncrement(QtCore.QSize(0, 0))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        self.line2 = QtWidgets.QFrame(self)
        self.line2.setGeometry(QtCore.QRect(830, 0, 31, 571))
        self.line2.setSizeIncrement(QtCore.QSize(0, 0))
        self.line2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line2.setObjectName("line")

        font = self.input_path_button.font()
        font.setWeight(QFont.Bold)
        self.input_path_button.setFont(font)
        self.reset_button.setFont(font)

        self.video_player = QLabel(self)
        self.video_player.setAlignment(Qt.AlignCenter)
        self.video_player.setGeometry(QtCore.QRect(10, 20, 420, 361))

        self.image1 = QLabel(self)
        self.image1.setAlignment(Qt.AlignCenter)
        self.image1.setGeometry(QtCore.QRect(860, 400, 200, 200))

        self.image2 = QLabel(self)
        self.image2.setAlignment(Qt.AlignCenter)
        self.image2.setGeometry(QtCore.QRect(1080, 400, 200, 200))

        self.video_player2 = QLabel(self)
        self.video_player2.setAlignment(Qt.AlignCenter)
        self.video_player2.setGeometry(QtCore.QRect(870, 20, 420, 361))

        # UI elements for displaying and editing video classification
        self.labelTable = QTableWidget(self)
        self.labelTable.setColumnCount(2)
        self.labelTable.setRowCount(100)
        self.labelTable.setHorizontalHeaderLabels(['Sports label', 'Timing(seconds)'])
        self.labelTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.labelTable.verticalHeader().setVisible(False)
        self.labelTable.setEditTriggers(QTableWidget.NoEditTriggers)
        # self.input_path_button.setGeometry(QtCore.QRect(290, 490, 75, 23))
        self.labelTable.setGeometry(QtCore.QRect(440, 20, 401, 361))

        # Put the UI elements together in a vertical layout
        # top_section = QWidget(self)
        # top_section_layout = QVBoxLayout(top_section)
        # top_section_layout.addWidget(self.input_path_textfield)
        # top_section_layout.addWidget(self.input_path_button)
        # top_section_layout.addWidget(self.video_player)
        # top_section.setLayout(top_section_layout)
        # top_section.setFixedWidth(100)

        # bottom_section = QWidget(self)
        # bottom_section_layout = QVBoxLayout(bottom_section)
        # bottom_section_layout.addWidget(self.labelTable)
        # bottom_section.setLayout(bottom_section_layout)

        # splitter = QSplitter(Qt.Vertical, self)
        # splitter.addWidget(top_section)
        # splitter.addWidget(bottom_section)

        # layout = QHBoxLayout(self)
        # layout.addWidget(splitter)
        # self.setLayout(layout)


    def reset_ui(self):
        self.video_player.clear()
        self.video_player2.clear()
        self.input_path_textfield.clear()
        self.labelTable.clearContents()
        self.timer.stop()
        self.reset_button.setEnabled(False)
        self.image1.clear()
        self.image2.clear()
    
    def set_item(self,row,column,item):
        self.labelTable.setItem(row,column,QtWidgets.QTableWidgetItem(str(item)))
        # self.labelTable.setItem(row, column, QTableWidgetItem("{:.2f}".format(float(item))))

        # print("yes")

    def select_video(self):
        # Use QFileDialog to prompt the user to select a video file
        self.reset_ui()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self,"Select Video File", "","Video Files (*.mp4 *.avi *.mkv *.mov *.flv *.wmv *.mpeg *.mpg *.3gp *.webm)", options=options)
        if file_path:
            self.input_path_textfield.setText(file_path)
            
            # ex3.set1()
            # self.labelTable.setItem(1,1,QtWidgets.QTableWidgetItem(str(self.item)))
            self.play_video(file_path)
            
            # self.labelTable.setItem(2,1,QtWidgets.QTableWidgetItem(str(self.item)))




    def play_video(self, file_path):
        self.reset_ui()
        self.cap = cv2.VideoCapture(file_path)
        # self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.display_frame)
        self.timer.start(10)  # Update frame every 10 milliseconds
        self.process_video(file_path)


    #initial method to display video then display labels
    
    def display_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QPixmap.fromImage(QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888))
            self.video_player.setPixmap(q_img.scaled(self.video_player.width(), self.video_player.height(), Qt.KeepAspectRatio))
        else:
            self.timer.stop()
            self.cap.release()
        
            

    #new mwthod to display labels along with video
    # def display_frame(self):

    #     if os.path.exists(r"D:\\Sports Video Classification"):
    #         base_dir = r"D:\Sports Video Classification\SportsVideoClassification"
    #     elif os.path.exists(r"D:\\"):
    #         base_dir = r"D:\SportsVideoClassification"
    #     else:
    #         base_dir=r"C:\SportsVideoClassification"

    #     ret, frame = self.cap.read()
    #     if ret:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         height, width, channel = frame.shape
    #         bytes_per_line = 3 * width
    #         q_img = QPixmap.fromImage(QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888))
    #         self.video_player.setPixmap(q_img.scaled(self.video_player.width(), self.video_player.height(), Qt.KeepAspectRatio))

    #     # Process the frame and update the labels
    #         frame = cv2.resize(frame, (244, 224)).astype("float32")
    #         mean=np.array([123.68,116.779,103.939][::1],dtype="float32")
    #         frame -= mean
    #         model=load_model(os.path.join(base_dir, "videoClassificationModel"))
    #         predictions = model.predict(np.expand_dims(frame, axis=0))[0]
    #         results = np.array(predictions).mean(axis=0)
    #         index = np.argmax(results)
    #         lb=pickle.loads(open(os.path.join(base_dir, "videoClassificationBinarizer.pickle"),"rb").read())
    #         label = lb.classes_[index]
    #         self.set_item(0, 0, label)
    #         self.set_item(0, 1, str(121))

    #     else:
    #         self.timer.stop()
    #         self.cap.release()



    

    # def play_video(self, file_path):
    #     # Use OpenCV to read the selected video file and play it in the video player
    #     cap = cv2.VideoCapture(file_path)
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         height, width, channel =frame.shape
    #         bytes_per_line = 3 * width
    #         q_img = QPixmap.fromImage(QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888))
    #         self.video_player.setPixmap(q_img.scaled(self.video_player.width(), self.video_player.height(), Qt.KeepAspectRatio))
    #         # self.video_player.setPixmap(q_img.scaled(self.video_player.width(), self.video_player.height(), Qt.KeepAspectRatio))   
    #     cap.release()
    #     QApplication.processEvents()
    #     self.process_video(file_path)
    
    def display(self,taken,frame,width,height):
        
        print("Function running")
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        print("Frame recieved")
        bytes_per_line=3*width
        q_img = QPixmap.fromImage(QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888))
        self.video_player.setPixmap(q_img.scaled(self.video_player.width(), self.video_player.height(), Qt.KeepAspectRatio))
        QApplication.processEvents()

    #to display summary of the video labels
    def display_summary(self):
    # Calculate the total time spent for each label
        self.label_times = {}
        generated=set()
        # self.current_label=None
        for row in range(self.labelTable.rowCount()):
            self.label = self.labelTable.item(row, 0)
    
            self.timing = self.labelTable.item(row, 1)
            if self.label and self.timing:
                self.label = self.label.text()
                self.timing = float(self.timing.text())
                time_seconds = self.timing
                # if self.current_label is None or self.current_label!=self.label:
                #     self.label_times[self.label] = time_seconds
                # else:
                #     self.label_times[self.label] = time_seconds
                #     self.current_label=self.label
                if self.current_label is None or self.current_label != self.label:
                    self.label_times[self.label] = time_seconds
                    if self.label not in generated:
                        print("New label generated:", self.label)
                        generated.add(self.label)
                        self.update_frame()
                    # self.update_frame()
                    self.current_label = self.label
                elif self.current_label==self.label:
                    self.label_times[self.label]=time_seconds

                    
                    # self.update_frame()
                # if self.current_label==None:
                #     self.image1.setPixmap(self.q_img.scaled(self.image1.width(), self.image1.height(), Qt.KeepAspectRatio))
                #     self.current_label=self.label
    # Display the summary in a message box
        self.summary_text = 'Summary:\n'
        summary_list=list(self.label_times.values())
        empty=[0]*len(summary_list)
        print("Summary:",summary_list)
        for i in range(len(summary_list)):
            if i==0:
                empty[i]=summary_list[i]
            else:
                empty[i]=summary_list[i]-summary_list[i-1]

        j=0
        print("Empty",empty)
        for self.label, self.timing in self.label_times.items():
            self.summary_text += f'{self.label}: {empty[j]:.2f} seconds\n'
            j+=1
        

        
    def update_frame(self):
        if self.image1.pixmap() is None:
            self.image1.setPixmap(self.q_img.scaled(self.image1.width(), self.image1.height(), Qt.KeepAspectRatio))
        elif  self.image2.pixmap() is None:
            self.image2.setPixmap(self.q_img.scaled(self.image2.width(), self.image2.height(), Qt.KeepAspectRatio))
        self.current_label=self.label





    def process_video(self,path):


        if os.path.exists(r"D:\\Sports Video Classification"):
            base_dir = r"D:\Sports Video Classification\SportsVideoClassification"
        elif os.path.exists(r"D:\\"):
            base_dir = r"D:\SportsVideoClassification"
        else:
            base_dir=r"C:\SportsVideoClassification"

        row=0
        column=0
        # ex2Object=SportsVideoClassification()
        model=load_model(os.path.join(base_dir, "videoClassificationModel"))
        lb=pickle.loads(open(os.path.join(base_dir, "videoClassificationBinarizer.pickle"),"rb").read())
        outputVideo1=os.path.join(base_dir,"output04.avi")
        mean=np.array([123.68,116.779,103.939][::1],dtype="float32")
        Queue=deque(maxlen=128)
        capture_video=cv2.VideoCapture(path)
        frame_rate = capture_video.get(cv2.CAP_PROP_FPS)

        writer=None
        (Width,Height)=(None,None)
        frame_no=1
        while True:
            (taken,frame)=capture_video.read()
            if not taken:
                break
            if frame is None:
                print("Video not supported")
                return
            if frame_no%288==0:
                if Width is None or Height is None:
                    (Width,Height)=frame.shape[:2]
                self.output=frame.copy()
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame=cv2.resize(frame,(244,224)).astype("float32")
                frame-=mean
                predictions=model.predict(np.expand_dims(frame,axis=0))[0]
                Queue.append(predictions)
                results=np.array(Queue).mean(axis=0)
                i=np.argmax(results)
                label1=lb.classes_[i]
                text="Sport is:{}".format(label1)
                # print(text)
                self.set_item(row,column,label1)
                column+=1
                # self.set_item(row,column,frame_no/frame_rate)
                seconds = capture_video.get(cv2.CAP_PROP_POS_MSEC) / 1000
                minutes = int(seconds / 60)
                seconds %= 60
                # self.set_item(row, column, {minutes:02f}:{seconds:02f})
                # self.set_item(row, column, "{:02d}:{:02d}".format(int(minutes), int(seconds)))


                self.set_item(row,column,capture_video.get(cv2.CAP_PROP_POS_MSEC)/1000)
                column=0
                row+=1

                
                # ex2Object.display(taken,frame,Width,Height)
            


                # q_img = QPixmap.fromImage(QImage(frame.data, Width, Height, 3*Width, QImage.Format_RGB888))
                # ex2Object.video_player2.setPixmap(q_img.scaled(ex2Object.video_player2.width(), ex2Object.video_player2.height(), Qt.KeepAspectRatio))



                # obj=SportsVideoClassification()
                # obj.item="Sport is Table Tennis"
                # print(text)
                cv2.putText(self.output,text,(45,60),cv2.FONT_HERSHEY_COMPLEX,1.25,(255,0,0),5)
    
                if writer is None:
                    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
                    writer=cv2.VideoWriter("outputVideo4",-1,30,(244,224),True)
                    if writer is None:
                        print("Video not supported!")
                        return
                writer.write(self.output)
                # cv2.imshow("Working",output)
                bytes_per_line = 3 * Width
                self.output = cv2.resize(self.output, (self.video_player2.width(), self.video_player2.height()))
                self.q_img = QPixmap.fromImage(QImage(self.output.data, self.output.shape[1], self.output.shape[0], self.output.strides[0], QImage.Format_RGB888))
                self.video_player2.setPixmap(self.q_img.scaled(self.video_player.width(), self.video_player.height(), Qt.KeepAspectRatio))
                self.display_summary()
                QApplication.processEvents()
                


                
                
                
                key=cv2.waitKey(1)&0xFF
                if key==ord("q"):
                    break
            frame_no+=1
        print("Finalizing")
        if writer is not None:
            writer.release()
        capture_video.release()
        self.reset_button.setEnabled(True)
        # self.display_summary()
        summary_box = QMessageBox()
        summary_box.setWindowTitle('Summary of input video')
        summary_box.setIcon(QMessageBox.Information)
        summary_box.setDefaultButton(QMessageBox.Close)
        summary_box.setText(self.summary_text)
        summary_box.exec_()
        

    
  

    
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SportsVideoClassification()
    window.show()
    sys.exit(app.exec_())

