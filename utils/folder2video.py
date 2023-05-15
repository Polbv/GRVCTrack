#write a function that takes a folder with images and outputs a video

# In[ ]:
import cv2
import os
import argparse
import os.path as osp
def parse_args():
    """
    args for fc testing.
    """
    #Config args
    parser = argparse.ArgumentParser(description='PyTorch SiamFC Tracking Test')
    
    
    parser.add_argument('--inputfolder', default=  '../Tracking/datasets/Helical diverters/Lab',type=str, help="folder containing the images to be converted to video")
    parser.add_argument('--outputfolder', default= '../Tracking/datasets/Helical diverters/Lab' , type=str,help="folder folder to contain the video")
    args = parser.parse_args()

    return args
def foldertovideo(args):
    #video_name="C:\\Users\\USER\\Desktop\\Lab_GRVC\\Tracking\\dataset\\video\\video.avi"
    #video_name="../Desktop/Lab GRVC/Tracking/dataset/video/video.avi"
    outoputdir= args.outputfolder
    print(osp.basename(args.outputfolder))
    video_path=osp.join(args.outputfolder, osp.basename(args.outputfolder) + '.avi')
    print(video_path)
    video_name= args.outputfolder
    image_folder= args.inputfolder
    images = [img for img in os.listdir(image_folder)]
    images.sort()
    #print(images)


    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_path, 0,30, (width,height))
    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        #print(image)
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()

def main():
    args = parse_args()
    print('Called with args:')
    print(args)
    foldertovideo(args)
    return

if __name__ == '__main__':
    main()

