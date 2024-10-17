# %%
import cv2
import numpy as np
import mediapipe as mp
import time 
import pyautogui

# %%
import tkinter as tk
from PIL import Image, ImageTk

# %%
mp_face_mesh=mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh=mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks = True,
    min_detection_confidence =0.5,
    min_tracking_confidence = 0.5)
'''
mp_drawing =mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1,circle_radius=1)
'''

# %%
lEye=1
lEyeIndex=[145,159,9,10] # >0.20 , < 0.35
allLEyeIndex=[33,246,161,160,159,158,157,173,133,153,154,155,145,144,163,7]
rEye=1
allREyeIndex=[362,398,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
rEyeIndex=[374,386,9,10]
mouth=0
mouthIndex=[12,15,94,197]
allMouthIndex=[76,184,74,73,72,11,302,303,304,408,306,307,320,404,315,16,85,180,90,77]
colorCode=[[0,0,0],[0,255,0]]


# %%
for i in range(len(colorCode)):
    print(colorCode[i])

# %%
def distance(landmark1,landmark2,width,height):
    x1=int(landmark1.x*height)
    y1=int(landmark1.y*width)
    x2=int(landmark2.x*height)
    y2=int(landmark2.y*width)
    return np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

# %%
#ids of mouth 12 15 94 197
#ids of left eye 145 159 9 10
#ids of right eye 374 386 9 10
def valueDistance(facial_landmark, id1,id2,id3,id4,x,y): #x,y is heigh and witdh
    return distance(facial_landmark.landmark[id1],facial_landmark.landmark[id2],x,y)/distance(facial_landmark.landmark[id3],facial_landmark.landmark[id4],x,y)

# %%
def changeModeDetect(type , currentValue, thresholdMin, thresholdMax):
    if type:
        if (currentValue < thresholdMin):
            return 0,1
        else:
            return type,0
    else:
        if (currentValue > thresholdMax):
            return 1,1
        else:
            return type,0

# %%
screen_width, screen_height = pyautogui.size()

# Hiển thị giá trị min và max của tọa độ con trỏ chuột
print(f"Tọa độ tối thiểu: (0, 0)")
print(f"Tọa độ tối đa: ({screen_width - 1}, {screen_height - 1})")

# %%
prevTime=0 # check fps
frameCount=0 
av1=0
av2=0 #just value distance of eye
distanceValueLEye=0
distanceValueREye=0
distanceValueMouth=0
predistanvalue=0
pyautogui.PAUSE = False
change1=0 #change of mouth
change2=0 #change of leye
change3=0 #change of reye
xNoseMode=0
yNoseMode=0
xPt=0
yPt=0
activeMouseMode=0
noseModeLocation=[-1,-1] #location of nose, value change after change mode
frameDelay=3 # average frame for smooth blink
radius=0 #radius for distance of nose 
task=0
#for task 2
locateX1=locateX2=locateY1=locateY2=0


cap = cv2.VideoCapture(0)
# set up tkinter
root = tk.Tk()
root.title("Video Stream")
root.attributes('-topmost', True)
root.geometry('256x256+1650+800')
label = tk.Label(root)
label.pack()

# Vòng lặp while để stream video
while True:
    try:
        # Đọc frame từ camera
        ret, frame = cap.read()

        if not ret:
            print("Không thể nhận diện được frame")
            break

        # Chuyển đổi frame từ BGR (OpenCV) sang RGB (Tkinter)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame=cv2.flip(frame,1)
        #frame=cv2.resize(frame,(256,256))
        #faceFrame=frame.copy()
        #faceFrame[:] = 255
        result=face_mesh.process(frame)
        if result.multi_face_landmarks:
            for facial_landmark in result.multi_face_landmarks:
                #draw landmark on face
                mp_drawing.draw_landmarks(
                    frame,
                    facial_landmark,
                    connections = mp_face_mesh.FACEMESH_TESSELATION,#FACEMESH_CONTOURS
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                #check the distance per frame
                ids=lEyeIndex
                distanceValueLEye = distanceValueLEye + valueDistance(facial_landmark,ids[0],ids[1],ids[2],ids[3],frame.shape[0],frame.shape[1])
                ids=rEyeIndex
                distanceValueREye = distanceValueREye + valueDistance(facial_landmark,ids[0],ids[1],ids[2],ids[3],frame.shape[0],frame.shape[1])
                ids=mouthIndex
                distanceValueMouth = distanceValueMouth + valueDistance(facial_landmark,ids[0],ids[1],ids[2],ids[3],frame.shape[0],frame.shape[1])

                frameCount = frameCount +1
                if frameCount == frameDelay:
                    frameCount=0

                    mouth, ok1= changeModeDetect(mouth,distanceValueMouth/frameDelay ,0.3,0.6)
                    distanceValueMouth=0
                    if ok1==1:

                        if mouth == 1:
                            if lEye==1 and rEye==1:
                                task=1
                                noseModeLocation=[facial_landmark.landmark[19].x,facial_landmark.landmark[19].y]
                                radius=distance(facial_landmark.landmark[19],facial_landmark.landmark[195],frame.shape[1],frame.shape[0])
                            elif lEye==0 and rEye==1:
                                task=2
                                noseModeLocation=[facial_landmark.landmark[19].x,facial_landmark.landmark[19].y]
                                radius=distance(facial_landmark.landmark[19],facial_landmark.landmark[195],frame.shape[1],frame.shape[0])
                            elif lEye==1 and rEye==0:
                                task=3
                                noseModeLocation=[facial_landmark.landmark[19].x,facial_landmark.landmark[19].y]
                                radius=distance(facial_landmark.landmark[19],facial_landmark.landmark[195],frame.shape[1],frame.shape[0])
                        elif mouth == 0:
                            task=0


                    lEye, ok2= changeModeDetect(lEye,distanceValueLEye/frameDelay ,0.08,0.2)
                    av1=distanceValueLEye/frameDelay
                    distanceValueLEye=0


                    rEye, ok3= changeModeDetect(rEye,distanceValueREye/frameDelay ,0.08,0.2)
                    av2=distanceValueREye/frameDelay
                    distanceValueREye=0

                    change1= change1 + ok1
                    change2= change2 + ok2
                    change3= change3 + ok3

                    if task==1:
                        xNoseMode=int(noseModeLocation[0]*frame.shape[1])
                        yNoseMode=int(noseModeLocation[1]*frame.shape[0])
                        pt= facial_landmark.landmark[19]
                        xPT=int(pt.x*frame.shape[1])
                        yPT=int(pt.y*frame.shape[0])
                        currentMouseX, currentMouseY = pyautogui.position()
                        if np.sqrt((xPT-xNoseMode)*(xPT-xNoseMode)+(yPT-yNoseMode)*(yPT-yNoseMode))/radius>=1:
                            currentMouseX=(currentMouseX+ round(xPT-xNoseMode,1)/radius*10)
                            if currentMouseX<0:
                                currentMouseX=0
                            if currentMouseX>screen_width:
                                currentMouseX=screen_width
                            currentMouseY=(currentMouseY+ round(yPT-yNoseMode,1)/radius*10)
                            if currentMouseY<0:
                                currentMouseY=0
                            if currentMouseY>screen_height:
                                currentMouseY=screen_height
                            pyautogui.moveTo(currentMouseX, currentMouseY)
                        
                        if lEye==0 and ok2==1:
                            pyautogui.click()
                        if rEye==0 and ok3==1:
                            pyautogui.click(button='right')

                    elif task==2:
                        #for moving mouse pointer
                        xNoseMode=int(noseModeLocation[0]*frame.shape[1])
                        yNoseMode=int(noseModeLocation[1]*frame.shape[0])
                        pt= facial_landmark.landmark[19]
                        xPT=int(pt.x*frame.shape[1])
                        yPT=int(pt.y*frame.shape[0])
                        currentMouseX, currentMouseY = pyautogui.position()
                        if np.sqrt((xPT-xNoseMode)*(xPT-xNoseMode)+(yPT-yNoseMode)*(yPT-yNoseMode))/radius>=1:
                            currentMouseX=(currentMouseX+ round(xPT-xNoseMode,1)/radius*10)
                            if currentMouseX<0:
                                currentMouseX=0
                            if currentMouseX>screen_width:
                                currentMouseX=screen_width
                            currentMouseY=(currentMouseY+ round(yPT-yNoseMode,1)/radius*10)
                            if currentMouseY<0:
                                currentMouseY=0
                            if currentMouseY>screen_height:
                                currentMouseY=screen_height
                            pyautogui.moveTo(currentMouseX, currentMouseY)
                        #choose the start of drag position
                        if lEye==0 and ok2==1:
                            locateX1=currentMouseX
                            locateY1=currentMouseY
                        #choose the end of drag position
                        if rEye==0 and ok3==1:
                            locateX2=currentMouseX
                            locateY2=currentMouseY
                            pyautogui.moveTo(locateX1,locateY1)
                            pyautogui.dragTo(locateX2, locateY2, duration=1)
                    elif task==3:
                        #for scrolling mouse
                        xNoseMode=int(noseModeLocation[0]*frame.shape[1])
                        yNoseMode=int(noseModeLocation[1]*frame.shape[0])
                        pt= facial_landmark.landmark[19]
                        xPT=int(pt.x*frame.shape[1])
                        yPT=int(pt.y*frame.shape[0])
                        currentMouseX, currentMouseY = pyautogui.position()
                        if np.sqrt((xPT-xNoseMode)*(xPT-xNoseMode)+(yPT-yNoseMode)*(yPT-yNoseMode))/radius>=1:
                            pyautogui.scroll(int( -(yPT-yNoseMode)/radius)*10)

                if task != 0: # just for drawing nose and circle
                    cv2.circle(frame,(xNoseMode,yNoseMode), 2 ,(0,0,255),-1)
                    cv2.circle(frame,(xPT,yPT), 2 ,(0,255,0),-1)
                    cv2.circle(frame, (xNoseMode,yNoseMode), int(radius), (0, 255, 0), thickness=1)
            #cv2.putText(frame, str('lEye:'+str(round(av1,2)) + ' '+ 'rEye:'+ str(round(av2,2))),(100,100),cv2.FONT_HERSHEY_DUPLEX,fontScale=0.5,color=(0,255,255),thickness=1)
        cv2.putText(frame, 
                    #str(frame.shape)
                    'task:'+str(task)
                    ,(10,20),cv2.FONT_HERSHEY_DUPLEX,fontScale=1,color=(0,255,255),thickness=1)
        #frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #for drawing color of eye and mouth
        #leye
        for i in range(len(allLEyeIndex)):
            pt= facial_landmark.landmark[allLEyeIndex[i]]
            xPT=int(pt.x*frame.shape[1])
            yPT=int(pt.y*frame.shape[0])
            cv2.circle(frame,(xPT,yPT), 2 ,(colorCode[lEye]),-1)
        
        #reye
        for i in range(len(allREyeIndex)):
            pt= facial_landmark.landmark[allREyeIndex[i]]
            xPT=int(pt.x*frame.shape[1])
            yPT=int(pt.y*frame.shape[0])
            cv2.circle(frame,(xPT,yPT), 2 ,(colorCode[rEye]),-1)
        
        #mouth
        for i in range(len(allMouthIndex)):
            pt= facial_landmark.landmark[allMouthIndex[i]]
            xPT=int(pt.x*frame.shape[1])
            yPT=int(pt.y*frame.shape[0])
            cv2.circle(frame,(xPT,yPT), 2 ,(colorCode[mouth]),-1)
        
        faceFrame=frame.copy()
        faceFrame=cv2.resize(faceFrame,(256,256))
        img = Image.fromarray(faceFrame)
        imgtk = ImageTk.PhotoImage(image=img)

        label.imgtk = imgtk
        label.configure(image=imgtk)

        root.update()

    except Exception as e:
        break
cap.release()
cv2.destroyAllWindows()



