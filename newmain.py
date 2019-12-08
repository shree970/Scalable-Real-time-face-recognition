

#importing all libraries
#tkinter= for GUI
#PIL = Python Imaging Library
#cv2=opencv


from tkinter.ttk import *
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from utils.align_custom import AlignCustom
from utils.face_feature import FaceFeature
from utils.mtcnn_detect import MTCNNDetect
from utils.tf_graph import FaceRecGraph
import argparse
import sys
import json
from time import strftime, gmtime
import datetime
import csv
import time


#unimportant parameters-----------------------------------
global last_frame
last_frame = np.zeros((480, 640, 3), dtype=np.uint8)

#initialise camera feed
global cap


global det_peeps
det_peeps=0
global count
count=0



global member
#--------------------------------------------------

#set camera index: 0 for default webcame
cap = cv2.VideoCapture(0)


#GUI stuff ------------------

def retrieve_input(textBox,inp):
    global member
    member=textBox.get("1.0","end-1c")
    inp.destroy()
    inp.quit()

#-------------------------




#This section of code deals with face registration of a user:
#It takes a live image feed frame , finds face, crops and aligns, pass cropped frame into model to get features,
#,stores these 128-D feature embedding into txt file
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def register():
	# count=0

	#Pre-processing variables, initialise GUI
#--------------------------------------------------------------------------------------------------
	recval=0
	click = True

	inp=Tk()
	info=Label(inp, height=5, width=30,text="Enter your your full name",font=("Helvetica bold", 20))
	textBox=Text(inp, height=1, width=20,font=("Helvetica bold", 25))
	info.pack()
	textBox.pack()
	buttonCommit=tk.Button(inp, height=4, width=15, text="Submit",command=lambda:retrieve_input(textBox,inp),activebackground="black", activeforeground="cyan", bd=4, bg="#123d63", fg="gold", font=("Helvetica bold", 15))
	#command=lambda: retrieve_input() >>> just means do this when i press the button

	buttonCommit.pack()
	mainloop()
	global member
	print(member)
#--------------------------------------------------------------------------------------------------



	#store name of person in variable full_name

	full_name = member


	#if clicked, start registration proces

	while (click):


		#initialise data and time files, make sure there's '{}' already present in files, otherwise it will throw json error
		
#--------------------------------------------------------------------------------------------------

		f = open('./facestored.txt','r');
		face_data = json.loads(f.read());
		t=open('./timedata.txt','r');
		time_data=json.loads(t.read());

		#initialise features
		person_imgs = {"Center": []};
		person_features = {"Center": []};
		person_time={"Hour":[],"Min":[],"Sec":[]};

#--------------------------------------------------------------------------------------------------

		#Start camera

		while True:

#--------------------------------------------------------------------------------------------------

			#PART 1: FIND FACE AND ALIGN
			_,frame = cap.read(); 	#read one frame

		#TO DETECT FACE: call detect_face funtion from mtcnn_detect.py=face_detect

			rects, landmarks = face_detect.detect_face(frame, 80);  # min face size is set to 80x80, store and

		#--------------------------------------------------------------------------------------------------

			for (i, rect) in enumerate(rects):

				cv2.rectangle(frame,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(255,0,0), 6) #draw bounding box for the face

		#FOR ALIGNMENT: call align funtion from align_custom.py=aligner to perform affine transformation

				aligned_frame, pos = aligner.align(160,frame,landmarks[i]);

		#--------------------------------------------------------------------------------------------------

				if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
					if(pos=="Center"):									#we take only center of face when present for convinience to user
						person_imgs[pos].append(aligned_frame)
			cv2.imshow("Register Face",frame)

		#--------------------------------------------------------------------------------------------------




			#cv2.imwrite('./captured/'+full_name+'.jpg', frame) 		#save image, not needed

			key = cv2.waitKey(1) & 0xFF


			#PART 2: GET FEATURES
		#--------------------------------------------------------------------------------------------------

		#extract features: call get_features function from face_feature.py=extract_features

			for pos in person_imgs:
				person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]
		#--------------------------------------------------------------------------------------------------


			#POST PROCESSING
			#getting date and time of registration
		#--------------------------------------------------------------------------------------------------

			final_name=full_name
			#print(final_name)
			face_data[final_name] = person_features;
			currh=datetime.datetime.now().strftime('%H')
			currm=datetime.datetime.now().strftime('%M')
			currs=datetime.datetime.now().strftime('%S')
			person_time["Hour"]=currh
			person_time["Min"]=currm
			person_time["Sec"]=currs
			time_data[final_name]=person_time;

		#--------------------------------------------------------------------------------------------------


		#PART 3: SAVING THE FEATURES IN FILE


			t = open('./timedata.txt', 'w');
			t.write(json.dumps(time_data))
			f = open('./facestored.txt', 'w');
			f.write(json.dumps(face_data))

		#--------------------------------------------------------------------------------------------------



			print("Welcome ",final_name)
			print(" ")


			#MORE POST PROCESSING FOR GUI

			#--------------------------------------------------------------------------------------------------

			while(True):

				img = np.zeros((800,1400,3), np.uint8)
				#cv2.line(img,(0,0),(511,511),(255,0,0),5)
				font = cv2.FONT_HERSHEY_COMPLEX
				cv2.putText(img,"Welcome to the event:)",(50,300), font, 3,(47, 160, 181),6,cv2.LINE_AA)
				cv2.putText(img,final_name,(50,500), font, 3,(226, 185, 61),6,cv2.LINE_AA)
				cv2.imshow("Press 'q' to close window", img)
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					cv2.destroyAllWindows()
					break

			#--------------------------------------------------------------------------------------------------

			t.close()
			recval=0
			global count
			count=count+1
			click = False
			cv2.destroyAllWindows()
			break;





#This section of code deals with inference
#Same pipeline till getting features: compares features by calling find_people() according to the set threhold
#check if face is in database or not.\
#If person is found, checks if recognition is constant till 40 frames and outputs a message, for unkown face checks till 20 frames
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



def detect():
	print("Detecting People")

    #Initializing variables and flags
    #-------------------------------------------------------------------------------------------------------#

	det_peeps=0
	detect=0
	recog_data=[('', 0)]
	prev = ''
	flag=0
	findface=1
	counterface=0
	unknowncounter=0
    #-------------------------------Part 1: To Detect Face--------------------------------------------------------------#
	while(True):

        #Read one frame
		ret,frame=cap.read()

		#TO DETECT FACE: call detect_face funtion from mtcnn_detect.py=face_detect
		rects, landmarks = face_detect.detect_face(frame,80);#min face size is set to 80x80
		aligns = []
		positions = []

		for (i, rect) in enumerate(rects):

		#----------------------ALIGNMENT-----------------------------------#
    	#FOR ALIGNMENT: call align funtion from align_custom.py=aligner to perform affine transformation
			aligned_face, face_pos = aligner.align(160,frame,landmarks[i])


			if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
				aligns.append(aligned_face)
				positions.append(face_pos)
			else:
				print("Align face failed")

		#Checking if there is a face present
		if(len(aligns) != findface and len(aligns)== 0):

			prev= ''
			counterface=0
			unknowncounter=0

		findface=len(aligns)

	#---------------------------Part 2: If Face is present get FEATURES------------------#
		if(len(aligns) == 1 and face_pos=='Center') :


		#extract features: call get_features function from face_feature.py=extract_features

			features_arr = extract_feature.get_features(aligns)

		#--------Compare the embeddings in findPeople() present in newmain.py(This file)--------------#
			recog_data = findPeople(features_arr,positions);  #Pass the feature obtained from face_feature class, get_features function
			per = recog_data[0][0] #Name of person


			#--------------If the recog_data[0][0] i.e. the person is same increment counterface---------#

			if(recog_data[0][0] == prev and recog_data[0][0]!='Unknown'):
				findface=1
				counterface=counterface+1
                # counterface is used to count if the same person is present for 40 frames or not
                #DELETE THIS RUN INDEFINITELY
				if counterface==40:
					while(True):
					#-----------------------GUI output-------------------#
						img = np.zeros((800,1400,3), np.uint8)
						#cv2.line(img,(0,0),(511,511),(255,0,0),5)
						font = cv2.FONT_HERSHEY_COMPLEX
						cv2.putText(img,"Welcome to the event:)",(50,300), font, 3,(47, 160, 181),6,cv2.LINE_AA)
						cv2.putText(img,prev,(50,500), font, 3,(226, 185, 61),6,cv2.LINE_AA)
						cv2.imshow("Press 'q' to close window", img)
						key = cv2.waitKey(1) & 0xFF
						if key == ord("q"):
							cv2.destroyAllWindows()
							#storing recognised people names in text file
							a = open("attend.txt", "a");
							a.write(prev)
							break
						#-----------------------GUI output stuff-------------------#
					cv2.destroyAllWindows()
					counterface=0
					break




                    # IF no face is detected increment unknowncounter


			if(recog_data[0][0] == prev and recog_data[0][0] == 'Unknown'):
				findface=1
				counterface=0
				unknowncounter=unknowncounter+1

                #If there are 20 frames with Unknown person---------------------------


				if unknowncounter==20:
					while(True):
                        #-----------------------------GUI----------------------------#
						img = np.zeros((800,1400,3), np.uint8)
						#cv2.line(img,(0,0),(511,511),(255,0,0),5)
						font = cv2.FONT_HERSHEY_COMPLEX
						cv2.putText(img,"Sorry!, please register again:(",(50,300), font, 2,(47, 160, 181),6,cv2.LINE_AA)
						#cv2.putText(img,prev,(50,500), font, 3,(226, 185, 61),6,cv2.LINE_AA)
						cv2.imshow("Press 'q' to close window", img)
						key = cv2.waitKey(1) & 0xFF
						if key == ord("q"):
							cv2.destroyAllWindows()
							break
						#-------------GUI-----------------------
					cv2.destroyAllWindows()
					counterface=0
					
					#DELETE THIS IF DON'T WANT REGISTER AGAIN
					register()
					break




#Assign the name to prev, here next frame is assigned

			prev = recog_data[0][0]

			#print(unknowncounter)
			#print(unknowncounter)


			#Opencv display boudning box with names------------------------------------
			for (i,rect) in enumerate(rects):
				cv2.rectangle(frame,(rect[0],rect[1]),(rect[0] + rect[2],rect[1]+rect[3]),(0,255,0), 4)
				cv2.putText(frame,str(recog_data[0][0])+" - " + str(recog_data[0][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
			#Opencv display boudning box with names------------------------------------

		#exit---------------------------------------------------
		cv2.imshow("Detected People. Press q to exit",frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			cv2.destroyAllWindows()
			break




#this function does actual comparision of new and previous stored face using eucledian distance
#input features arrary outputs name of person or Unknown

def findPeople(features_arr, positions, thres = 0.6, percent_thres = 70):

    #Open File storing Embeddings
	f = open('./facestored.txt','r')
	face_data = json.loads(f.read());
	returnRes = [];
	for (i,features_128D) in enumerate(features_arr):
		result = "Unknown";
		smallest = sys.maxsize

    #Comparing Embeddings
		for person in face_data.keys():
			person_data = face_data[person][positions[0]];
			for data in person_data:
				distance = np.sqrt(np.sum(np.square(data-features_128D)))  #Calculating Distance
				if(distance < smallest):
					smallest = distance
					# print("Smallest",smallest)
					result = person;
		percentage =  min(100, 100 * thres / smallest)


#Compare with threshold
		if percentage <= percent_thres :
			result = "Unknown"
		returnRes.append((result,percentage))

        #Return name and Percentage
	return returnRes

def quit():
	root.quit()

if __name__ == '__main__':

	# Renaming of files for easier understanding
	#----------------------------------------------------------------------------------------------
	FRGraph = FaceRecGraph();
	aligner = AlignCustom();
	extract_feature = FaceFeature(FRGraph)
	face_detect = MTCNNDetect(FRGraph, scale_factor=2);

    #---------------------------------------GUI window --------------------------------------------#
	y=2
	root = Tk()
	root.title("Event Registration")
	root.resizable(width=False, height=False)
	cwgt=Canvas(root,width=1250,height=550,borderwidth=60,background='white',
                 relief='raised')
	cwgt.grid(row=0, column=0, sticky="nsew")
	cwgt.pack(expand=True, fill=BOTH)
	image1=PhotoImage(file="background2.png")
	cwgt.img=image1
	cwgt.create_image(0, 0, anchor=NW, image=image1)
	canvas_id = cwgt.create_text(200,500, anchor="nw")
	cwgt.itemconfig(canvas_id, text="REGISTER", width=780)
	cwgt.itemconfig(canvas_id, font=("Helvetica bold", 30),fill="gold",activefill="cyan")
	img1 = ImageTk.PhotoImage(Image.open("register.png"))  # PIL solution
	b1=Button(cwgt, text="REGISTER",command=register)
	b1.config(image=img1,width=230,height=230)
	button1_window = cwgt.create_window(10, 50, anchor=NW, window=b1)
	b1.place(x=190,y=250)
	b1.update()
	canvas_id1 = cwgt.create_text(600,500, anchor="nw")
	cwgt.itemconfig(canvas_id1, text="VERIFY", width=780)
	cwgt.itemconfig(canvas_id1, font=("Helvetica bold", 30),fill="gold")
	img = ImageTk.PhotoImage(Image.open("log1.jpeg"))  # PIL solution
	b2 = Button(cwgt,text = "button 2",command=detect)
	button2_window = cwgt.create_window(10, 50, anchor=NW, window=b2)
	b2.config(image=img,width=230,height=230)
	b2.place(x=560,y=250)
	b2.update()
	canvas_id2 = cwgt.create_text(1000,500, anchor="nw")
	cwgt.itemconfig(canvas_id2, text="EXIT", width=780)
	cwgt.itemconfig(canvas_id2, font=("Helvetica bold", 30),fill="gold")
	img2 = ImageTk.PhotoImage(Image.open("exit4.png"))
	b3 = Button(cwgt,text = "button 3",command=quit)
	button3_window = cwgt.create_window(10, 50, anchor=NW, window=b3)
	b3.config(image=img2,width=230,height=230)
	b3.place(x=920,y=250)
	b3.update()
	root.mainloop()
	#---------------------------------------GUI window --------------------------------------------#
cap.release()
