#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
#from torchvision.models.detection.anchor_utils import AnchorGenerator


# In[5]:


import numpy as np
from PIL import Image
from torchvision.transforms import transforms as transforms
import time


# In[6]:


# pairs of edges for 17 of the keypoints detected from try and error
edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6)
]
point = np.asarray(edges)
point_numbers = len(set(point.flatten()))


# In[7]:


def draw_keypoints_and_boxes(outputs, image,pullups=0):
    # the `outputs` is list which in-turn contains the dictionary 
    cor = []
    sucuess = False
    for i in range(len(outputs[0]['keypoints'])):
        # get the detected keypoints
        keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
        #print("keypoints",keypoints)
        # get the detected bounding boxes
        boxes = outputs[0]['boxes'][i].cpu().detach().numpy()
    
        # proceed to draw the lines and bounding boxes 
        if outputs[0]['scores'][i] > 0.8 and sucuess == False: # proceed if confidence is above 0.8
            sucuess=True
            
            #"Text"
            cv2.putText(image, str(pullups), (80,80), cv2.FONT_HERSHEY_SIMPLEX, 
                   2, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
            
            
            keypoints = keypoints[:, :].reshape(-1, 3)
            cor = [keypoints[:, :-1]]
            
            for p in range(keypoints.shape[0]):
                # draw the keypoints
                cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])), 
                            3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            # draw the lines joining the keypoints
            for ie, e in enumerate(edges):
                # get different colors for the edges
                rgb = matplotlib.colors.hsv_to_rgb([
                    ie/float(len(edges)), 1.0, 1.0
                ])
                rgb = rgb*255
                # join the keypoint pairs to draw the skeletal structure
                #print(e,(keypoints[e, 0][0], keypoints[e, 1][0]),
                #        (keypoints[e, 0][1], keypoints[e, 1][1]))
                cv2.line(image, (keypoints[e, 0][0], keypoints[e, 1][0]),
                        (keypoints[e, 0][1], keypoints[e, 1][1]),
                        tuple(rgb), 5, lineType=cv2.LINE_AA)
            
            # draw the bounding boxes around the objects
            
            cv2.rectangle(image, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])),
                          color=(0, 255, 0), 
                          thickness=10)
        else:
            continue
    return image,sucuess,cor 


# In[8]:


def get_model(min_size=800):
    # initialize the model
    # or keypointrcnn_resnet50_fpn
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                                   num_keypoints=point_numbers, 
                                                                   min_size=min_size)
    #backbone=torchvision.models.mobilenet_v2(pretrained=True).features
    #backbone.out_channels = 1280
    #anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))
    #roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7, sampling_ratio=2)
    #keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                                     output_size=14,sampling_ratio=2)
    #model = torchvision.models.detection.KeypointRCNN(backbone,num_classes=2,
    #                                                  rpn_anchor_generator=anchor_generator,
    #                                                  box_roi_pool=roi_pooler,
    #                                                    keypoint_roi_pool=keypoint_roi_pooler)
    
    return model


# In[9]:


# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])


# In[10]:


# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model = get_model().to(device).eval()


# In[11]:


# Test on single figure
args={}
args['input'] = "/home/sam/Dropbox/Camera Uploads/PXL_20210413_223143520.MP.jpg"
image_path = args['input']
image = cv2.imread(image_path)
#image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
# NumPy copy of the image for OpenCV functions
orig_numpy = np.array(image, dtype=np.float32)
# convert the NumPy image to OpenCV BGR format
orig_numpy = cv2.cvtColor(orig_numpy, cv2.COLOR_RGB2BGR) / 255.
# transform the image
image = transform(image)
# add a batch dimension
image = image.unsqueeze(0).to(device)
# get the detections, forward pass the image through the model
with torch.no_grad():
    outputs = model(image)
# draw the keypoints, lines, and bounding boxes
output_image,sucess,cor = draw_keypoints_and_boxes(outputs, orig_numpy)


# In[12]:


plt.imshow(output_image)


# In[13]:


outputs;


# In[14]:


cor;


# In[15]:


coordinate = outputs[0]['keypoints'][0].cpu().detach().numpy()
plt.scatter(coordinate[:,0],-coordinate[:,1])


# In[16]:


#!mkdir outputs
#args['input'] = "/home/sam/Dropbox/Camera Uploads/PXL_20210413_224318815.mp4"
args['input'] = 0 #webcam
args['min_size']=800
cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# get the video frames' width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# set the save path
save_path = "./outputs/real_time_"+str(args['min_size'])+".mp4"
# define codec and create VideoWriter object 
#w,h = 540,960
out = cv2.VideoWriter(save_path, 
                      cv2.VideoWriter_fourcc(*'mp4v'), 5, 
                      (400,frame_height))
print(frame_height,frame_width)
frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second


# In[16]:


# read until end of video

cors=[]
num_frame=0
frames=[]
state=-1 #-1 = down, 1 = up
pullups=0
initial_time =time.time()
delta = 3
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    frame = frame[:,:400,:]
    #print(frame.shape)
    #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #count+=1
    if ret == True and time.time()-initial_time>delta:
        #print(count)
        #print(frame)
        pil_image = Image.fromarray(frame).convert('RGB')
        orig_frame = frame
        # transform the image
        image = transform(pil_image)
        # add a batch dimension
        image = image.unsqueeze(0).to(device)
        # get the start time
        start_time = time.time()
        # get the detections, forward pass the frame through the model
        with torch.no_grad():
            output_image = model(image)
            #print("max score",np.max(output_image[0]['scores'].cpu().detach().numpy()))
        # get the end time
        end_time = time.time()
        output_image,sucess,cor = draw_keypoints_and_boxes(output_image, orig_frame,pullups)
        if(sucess==False):continue
        ys = np.mean(cor[0][1:5,1])
        th = np.mean(cor[0][7:9,1])
        #print(ys,th,state,(ys<th),(ys<th)*state,cor[0][7,1],cor[0][5,1])
        if((np.sign(ys-th)*state)==-1):
            pullups+=(state==1)*(cor[0][7,1]<cor[0][5,1])
            state*=-1
        num_frame+=1
        #print(num_frame)
        cors+=[cor]
        frames+=[frame]
        # get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1
        wait_time = max(40, int(fps/4))
        #ims = cv2.resize(output_image, (h,w))  
        #print(output_image.shape)
        cv2.imshow('Pose detection frame', output_image)
        out.write(output_image)
        
        #out.write(frame)
        
        # press `q` to exit
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    elif(ret == False):
        break


# In[17]:


#len(cors)


# In[18]:


#len(frames)


# In[19]:


plt.imshow(frames[30])


# In[20]:


cors=np.asarray(cors)
for cor in cors[30:31]:
    for i,p in enumerate(cor[0]):
        #print(p)
        plt.text(p[0],p[1],str(i))
    for e in edges:
        x = [cor[0][e[0],0],cor[0][e[1],0]]
        y = [cor[0][e[0],1],cor[0][e[1],1]]
        x=np.asarray(x)
        y=np.asarray(y)
        plt.plot(x,y,'--')
plt.xlim([0,320])
plt.ylim([640,0])


# In[21]:


ys = [np.mean(cor[0][1:5,1]) for cor in cors]
ys2 = [np.mean(cor[0][7:9,1]) for cor in cors]

plt.plot(ys,".")
plt.plot(ys2,".")

#plt.ylim([190,210])


# In[22]:


#ys


# In[23]:


# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
save_path = "./outputs/dummpy.mp4"
# define codec and create VideoWriter object 
#w,h = 540,960
out = cv2.VideoWriter(save_path, 
                      cv2.VideoWriter_fourcc(*'mp4v'), 20, 
                      (frame_width,frame_height))
#!rm ./outputs/dummpy.mp4


# In[ ]:




