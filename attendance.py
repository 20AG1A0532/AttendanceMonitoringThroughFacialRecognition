import cv2 
#It is an open-source library that can be used to perform tasks like face detection, objection tracking, landmark detection, and much more.

import numpy as np 
#NumPy is a library used for working with arrays(Faster than lists).

import face_recognition 
#The face_recognition module lets you recognize faces in a photograph or folder full for photographs.

import os 
#The OS module in Python provides functions for creating and removing a directory (folder), fetching its contents, changing and identifying the current directory, etc.

from datetime import datetime 
#The datetime module supplies classes for manipulating dates and times.

import sqlite3 
#Python SQLite3 module is used to integrate the SQLite database with Python.

con=sqlite3.connect("Attendance.db") 
#Creates/Connects the python file with the database(sqlite3).

cur = con.cursor() 
#In order to execute SQL statements and fetch results from SQL queries, we will need to use a database cursor. Call con.cursor() to create the Cursor.

cur.execute("CREATE TABLE AttendanceList(SNAME text,Etime text,Edate text)") #Creating a Table.

path = 'images' 
images = []
studentNames = []
myList = os.listdir(path) #Saving all files of the directory into the list.
for cu_image in myList:
    current_Image = cv2.imread(f'{path}/{cu_image}')
    images.append(current_Image)
    studentNames.append(os.path.splitext(cu_image)[0])


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def attendance(name):
    with open('Attendance.csv', 'r+',encoding='cp856') as f:
        DataList = f.readlines()
        nameList = []
        for line in DataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'{name},{tStr},{dStr}\n')
            cur.execute("""INSERT INTO AttendanceList (SNAME,Etime,Edate) VALUES( ?,?,? )""",(str(name),str(tStr),str(dStr))) #Insert into table.
            con.commit() #Commit the changes into the database.


encodeListArrays = faceEncodings(images)
print('All Encodings Completed')

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    faces = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
    faces = faces[:, :, ::-1]
    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListArrays, encodeFace)
        faceDis = face_recognition.face_distance(encodeListArrays, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = studentNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Camera', frame)
    if cv2.waitKey(2) == 13 :
      break

con.close() #at last disconnect python file from the database
vid.release()
cv2.destroyAllWindows() 