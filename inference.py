import torch 
import cv2 
import numpy as np 
import argparse 
import os 
import pandas as pd 

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default=None)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--size', type=int, default=640)
parser.add_argument('--version', type=int, default=None)

args = parser.parse_args()

source = args.source 
device = args.device
size = args.size
version = args.version

if source is None: 
    raise ValueError('Please specify source')

if version is None:
    model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
else:
    runs_path = os.path.join('yolov5', 'runs', 'train')
    runs = os.listdir(runs_path)
    if version == -1:
        latest_run = runs[version]
    else:
        latest_run = runs[version-1]
    model = torch.hub.load('yolov5', 'custom', path=os.path.join(runs_path, latest_run, 'best.pt'), source='local') 
    
wheel = cv2.imread(os.path.join('resources', 'steering_wheel_image.jpg'))   
wheel = cv2.cvtColor(wheel, cv2.COLOR_BGR2GRAY)
rows, cols = wheel.shape
    
if source is not None:
    cap = cv2.VideoCapture(source)
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640,640))
            res = model(frame)
            img = np.squeeze(res.render())
            res_df = pd.concat(res.pandas().xyxy)
            centriod = res_df[['xmin', 'xmax']].mean(axis=1).mean()
            h = (320-centriod)/320
            h = (h+1)*(h)*(h-1)
            rot = h*630
            cv2.imshow('Result', np.squeeze(res.render()))
            M = cv2.getRotationMatrix2D((cols/2,rows/2), rot, 1)
            dst = cv2.warpAffine(wheel, M, (cols,rows))
            cv2.imshow('Steering wheel', dst)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break  
        else: 
            break
    cv2.destroyAllWindows()
        
    