import os.path as osp
import os
import cv2
import argparse
def parse_args():
    """
    args for fc testing.
    """
    #Config args
    parser = argparse.ArgumentParser(description='PyTorch SiamFC Tracking Test')
    
    
    parser.add_argument("--inputfolder", default=  "../Tracking/datasets/Helical diverters/Lab/rgb", help="folder containing the images to be converted to video")
    
    args = parser.parse_args()

    return args
def img_order(args):
    
    new_path_name=osp.join(args.inputfolder,'ordered')
    try:
        os.mkdir(new_path_name)
    except:
        pass
    images=os.listdir(args.inputfolder)

    print(images)
    N=len(images)
    for i in range(N):
        j=("{:05d}".format(i))

        new_path=osp.join(new_path_name,j + '.jpg')
        #print(new_path)

        p=osp.join(args.inputfolder,'rgb_'+str(i) + '.jpg')
        print(p)
        im=cv2.imread(p)

        cv2.imwrite(new_path,im)
def main():
    args = parse_args()
    #path="../Desktop/Lab GRVC/Tracking/dataset/rgb"
    img_order(args)


if __name__ == "__main__":
    main()


