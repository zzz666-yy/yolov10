import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    # model = YOLO('ultralytics/cfg/models/v5/improved-yolov5.yaml')
    model = YOLO('runs/train/yolo v10n-MAFPN-MAN-SimAM(FLoW_IMG)/weights/best.pt') # select your model.pt path
    model.predict(source='D:/project/dataset/FLoW_IMG dataset/images/detect',
                  imgsz=640,
                  project='runs/detect',
                  name='yolo v10n-MAFPN-MAN-SimAM(FLoW_IMG)',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  line_width=2, # line width of the bounding boxes
                  show_conf=True, # do not show prediction confidence
                  show_labels=True, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )