from src.detector import detect_faces
from PIL import Image,ImageDraw
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
@torch.no_grad()
def img_locate(path):
    img=Image.open(path)
    boundboxes,landmark=detect_faces(img.copy())
    print(boundboxes,landmark)
    img_draw=ImageDraw.Draw(img)
    print(img.size,type(landmark))
    for i,box in enumerate(boundboxes):
        box=np.array(box)
        lm=np.array(landmark[i],np.int32)
        fill=(0,0,255)
        for j in range(0,len(lm)//2):
            print('j:{}'.format(j))
            img_draw.point((lm[j],lm[j+5]),fill=fill)
        print('box:{}'.format(box))
        img_draw.rectangle(tuple(box[:4].astype(np.int32)),outline=(255,0,0), width=2)
        img_draw.text(tuple(box[:2].astype(np.int32)),text="{}".format(box[-1]),fill=fill)
    img.show()
    plt.show()
    #img_draw.rectangle(label[:4], outline=(255, 0, 0), width=0)
    #img_draw.text(label[:2], text=str(label[5]), fill=(255, 0, 0), font=font)
    #img.show()
@torch.no_grad()
def mtcnn_crop(in_path,out_path,crop_size=(112,96)):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    imgs_folder=os.listdir(in_path)
    for img_folder in tqdm.tqdm(imgs_folder):
        if not os.path.exists(os.path.join(out_path,img_folder)):
            os.makedirs(os.path.join(out_path,img_folder))
        img_names=os.listdir(os.path.join(in_path,img_folder))
        for name in img_names:
            img=Image.open(os.path.join(in_path,img_folder,name))
            boundboxes, landmark = detect_faces(img)
            index=0
            score=boundboxes[0][-1]
            for i,box in enumerate(boundboxes):
                if(box[-1]>score):
                    index=i
                    score=box[-1]
            box=boundboxes[index][:4].astype(np.int32)
            img_crop=img.crop(box).resize(crop_size,Image.BICUBIC)
            img_crop.save(os.path.join(out_path,img_folder,name))


if __name__ == '__main__':
    # path=r'G:\FaceRetrieval\lfw_funneled\Jan_Peter_Balkenende'
    # name='Jan_Peter_Balkenende_0001.jpg'
    # img_locate(os.path.join(path,name))
    in_path=r'G:\FaceRetrieval\lfw_funneled'
    out_path=r'G:\FaceRetrieval\lfw_funneled_croped'
    mtcnn_crop(in_path,out_path)
