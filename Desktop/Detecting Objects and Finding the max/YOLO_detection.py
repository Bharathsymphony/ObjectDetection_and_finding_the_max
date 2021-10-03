import cv2
import numpy as np
import matplotlib.pyplot as plt

yolo=cv2.dnn.readNet("yolov3.weights","yolov3.cfg")

#Initializing required Lists
class_names=[]
images=[]
no_of_boxes=[]
timestamps=[]

#Getting objects that YOLO acan detect
with open("coco_classes.names","r") as file:
    class_names=[line.strip() for line in file.readlines()]
       
layer_names=yolo.getLayerNames()
out_layer=[layer_names[i[0]-1] for i in yolo.getUnconnectedOutLayers()]

#Reading Video 
cap=cv2.VideoCapture("Sample_video.mp4")

while True:
    _,img=cap.read()
    height,width,_=img.shape

#Detecting the Objects
    blob=cv2.dnn.blobFromImage(img,0.0039,(416,416),(0,0,0),True,crop=False)

    yolo.setInput(blob)
    outputs=yolo.forward(out_layer)

    class_ids=[]
    confidences=[]
    boxes=[]

    for output in outputs:
        for detect in output:
            scores=detect[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            
            if confidence >0.5:
                center_x=int(detect[0]*width)
                center_y=int(detect[1]*height)

                w=int(detect[2]*width)
                h=int(detect[3]*height)

                x=int(center_x-w/2)
                y=int(center_y-h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

#Drawing Rectangles and Lebeling the object
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h=boxes[i]
            label=str(class_names[class_ids[i]])
            conf_label=str(round(confidences[i],2))
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            cv2.putText(img,label+" "+conf_label,(x,y+10),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)

#Storing the Image,Number of boxes and Timestamp to list
    cv2.imshow("Timestamp_of_Video",img)
    key=cv2.waitKey(1)
    images.append(img)
    no_of_boxes.append(len(boxes))
    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    
    if key==27:
#Getting the Timestamp with maximum number of objects and saving the Timestamp         
        max_value=max(no_of_boxes)
        max_index=no_of_boxes.index(max_value)
        timestamp_of_max=timestamps[max_index]
        cv2.imwrite(str(timestamp_of_max)+"ms.jpg",images[max_index])
        plt.imshow(images[max_index])
        break
    
cap.release()
cv2.destroyAllWindows()

