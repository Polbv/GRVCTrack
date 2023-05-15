from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer
from array import array
import cv2
import os
import argparse
import torch
import numpy as np
import time
import glob
from PIL import Image
import os.path as osp

def parse_args():
    """
    args for fc testing.
    """
    #Config args
    parser = argparse.ArgumentParser(description='PyTorch SiamFC Tracking Test')
    
    parser.add_argument('--device', default="cpu", type=str, help='inference device')
    parser.add_argument('--trt', default=None,  help='enable trt boosting')
    parser.add_argument('--video', default=True, type=str, help='test a video in benchmark')
    parser.add_argument('--input', default='videos/palace.mp4', help='image or video file to be tracked')
    parser.add_argument("--cfg", default="./Weights/yolov4-aerialcore.cfg", help="path to config file")
    parser.add_argument("--data_file", default="./Weights/coco.names", help="path to data file")
    parser.add_argument("--weights", default="./Weights/yolov4-aerialcore_best.weights", help="yolo weights path")
    parser.add_argument("--output_dir", default="/output", help="output folder")
    parser.add_argument("--save_result", default=False, help="whether to save result")
    parser.add_argument('--vis', default=True, help='visualize tracking results')
    #detection args
    
    parser.add_argument("--thresh", type=float, default=0.7, help="remove detections with lower confidence")
    parser.add_argument("--iou", type=float, default=.85, help="min iou")
  
    args = parser.parse_args()

    return args
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def detection(args,im,model):
    labels = open(args.data_file).read().strip().split('\n')
    timer2=Timer()
    assert im is not None
    scale = 0.00392
    timer2.tic()
    blob=cv2.dnn.blobFromImage(im,scale,(416,416),(0,0,0),True,crop=False)#size is given by the model cfg file and weigths
    model.setInput(blob)
    model.enableWinograd(True)#check if it's possible to use winograd first
    outputs=model.forward(get_output_layers(model))
    boxes=[]
    confidences=[]
    classIDs=[]
    nms_threshold=0.5
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    #if args.trt is not None
        #from TensorRT import Tensor
    for output in outputs:
        for detection in output:
            scores=detection[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>args.thresh:
                
                center_x=int(detection[0]*im.shape[1])
                center_y=int(detection[1]*im.shape[0])
                w=int(detection[2]*im.shape[1])
                h=int(detection[3]*im.shape[0])
                x=int(center_x-w/2)
                y=int(center_y-h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                classIDs.append(class_id)
    timer2.toc()
    print("detections done",1. / timer2.average_time)
   
    indices = cv2.dnn.NMSBoxes(boxes, confidences, args.thresh,nms_threshold)
    print("nms done")
    boxes_nms=[]
    confidences_nms=[]
    classIDs_nms=[]
    for i in indices:
        boxes_nms.append(boxes[i])
        confidences_nms.append(confidences[i])
        classIDs_nms.append(classIDs[i])


        #color = [int(c) for c in colors[classIDs[i]]]
        #x,y,w,h=boxes[i]
        #cv2.rectangle(im,(x,y), (x+w,y+h), color, 2)
        #text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
        #cv2.putText(im, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    
    return(boxes_nms,confidences_nms,classIDs_nms,im)
               


class parametros:
    demo="video"
    
    path="./videos/palace.mp4"
    camid=0
    save_result=True
    ckpt=None
    device="cpu"
    conf=None
    nms=None
    tsize=None
    fps=30
    fp16=False
    fuse=True
    trt=False
    track_thresh=0.5
    track_buffer=30
    match_thresh=0.8
    aspect_ratio_thresh=1.6
    min_box_area=10
    mot20=False
def track(args,tracker,image,id,model):
   
    timer=Timer()
    #init variables
   
   
    
    timer.tic()
    bboxes,confidences,class_ids,rgb_image=detection(args,image,model)
    
        
    #convertir a formato de deteccion de yolo a formato de deteccion de bytetrack
    bboxes=xywh_to_x1y1x2y2(bboxes)  
    n=len(bboxes)
    detections=np.zeros((n,5))
    i=0
    for b in bboxes:
        detections[i,0:4]=b
        detections[i,4]=confidences[i]
        i=i+1
       

    just_dect = False#wheter to track or not
    if not just_dect:
        

        results = []
        online_targets=tracker.update(detections,rgb_image.shape,rgb_image.shape)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        j=0
        for targets in online_targets:
            tlwh=targets.tlwh
            tid=targets.track_id
            tscore=targets.score
          
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(tscore)
            results.append([tid,tlwh[0],tlwh[1],tlwh[2],tlwh[3],tscore])
            j=j+1 
        print(j,":tracks found") 
                            
                            
        timer.toc()
        image = plot_tracking(
            image, online_tlwhs, online_ids, frame_id=id, fps=1. / timer.average_time
            )
            
    return image,results
def xywh_to_x1y1x2y2(bbox):
    bboxes=[]
    for i in bbox:
        x1 = i[0]
        y1 = i[1]
        x2 = i[0] + i[2]
        y2 = i[1] + i[3]
        bboxes.append([x1, y1, x2, y2])

    return bboxes
def main(args):
    
    
    pars=parametros()
    model=cv2.dnn.readNet( args.weights,args.cfg)

    if args.video:
        pars.demo="video"
        tracker=BYTETracker(pars,pars.fps)
        stream=cv2.VideoCapture(args.input)
        #stream=cv2.VideoCapture(0)
        assert stream is not None
        frame_id=0

        vis_path=osp.join(args.output_dir,osp.basename(args.input),"vis")
        
        while stream.isOpened():
            
            ret,frame=stream.read()
            if not ret:
                break
            
            print("tracking frame number:",frame_id)

            tracked_im,results=track(args,tracker,frame,frame_id,model)

            if args.vis:
                cv2.imshow("window",tracked_im)
                vis=1


            if args.save_result:
                if frame_id==0:
                    frame_size=(tracked_im.shape[1],tracked_im.shape[0])
                    print("frame size:",frame_size)
                    vid_writer = cv2.VideoWriter('args.output_dir'+'.avi',cv2.VideoWriter_fourcc(*'MPEG'), 30.0,frame_size)
                else:
                    vid_writer.write(tracked_im)


            frame_id=frame_id+1

            if vis==1 & cv2.waitKey(25) & 0xFF==ord('q'):
                break


        if args.vis:    
            stream.release()
            cv2.destroyAllWindows()

    if not args.video:
        pars.demo="image"
        tracker=BYTETracker(pars,pars.fps)
        frame_id=0
        frame=cv2.imread(args.input)
        tracked_im,results=track(args,tracker,frame,frame_id)
        if args.vis:
            cv2.imshow("tracked_im",tracked_im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if args.save_result:
            im_path=osp.joinpath(args.output,args.input)
            cv2.imwrite(im_path,tracked_im)
    #print(results)

    

if __name__ == "__main__":
    args = parse_args()
    main(args)