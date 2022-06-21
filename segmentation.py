import torch
import cv2
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features 
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

model.load_state_dict(torch.load("best_weights.torch"))
model.eval()

def apple_segmentation(image, class_target):

  imageSize = (600, 600)
  images = cv2.resize(image, imageSize, cv2.INTER_LINEAR)
  images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)
  images = images.swapaxes(1, 3).swapaxes(2, 3)

  with torch.no_grad():
    pred = model(images)

  im= images[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)
  im2 = im.copy()

  best_segmentation = 0
  best_mask = 0

  for i in range(len(pred[0]['masks'])):
      msk=pred[0]['masks'][i,0].detach().cpu().numpy()
      scr=pred[0]['scores'][i].detach().cpu().numpy()
      if scr > 0.8:
          if class_target == 0:
            im2[:,:,0][msk>0.5] = 0 # random.randint(0,255)
            im2[:, :, 1][msk > 0.5] = 255 # random.randint(0,255)
            im2[:, :, 2][msk > 0.5] = 0 # random.randint(0, 255)
          if class_target == 1:
            im2[:,:,0][msk>0.5] = 0 # random.randint(0,255)
            im2[:, :, 1][msk > 0.5] = 0 # random.randint(0,255)
            im2[:, :, 2][msk > 0.5] = 255 # random.randint(0, 255)

  bounding_box = pred[0]['boxes'].detach().cpu().numpy()[np.argmax(pred[0]['scores'].detach().cpu().numpy())]

  cv2.rectangle(im2, (int(bounding_box[0]), int(bounding_box[1])), (int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3])), (255, 0, 0), 2)
  print(str(round(np.max(pred[0]['scores'].detach().cpu().numpy()), 2 )) + ' %')
  cv2.putText(im2, str(round(np.max(pred[0]['scores'].detach().cpu().numpy()), 2 )) + ' %', 
                    (int(bounding_box[0]), int(bounding_box[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1,
                    (255, 0, 0), 
                    2, cv2.LINE_AA)
  
  return im2

if __name__ == "__main__":
  image = cv2.imread('apples/image_3.jpg', cv2.COLOR_BGR2RGB)
  cv2.imshow('output', apple_segmentation(image, 0))
  cv2.waitKey(0)
  cv2.destroyAllWindows()