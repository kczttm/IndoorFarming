python train.py --batch-size 32 --epochs 50 --data data2.yaml --weights /Users/tonytu/Desktop/Soft_Robotics_Internship/YOLO_UR5_Test/yolov5/runs/train/exp7/weights/last.pt

python val.py --weights /Users/tonytu/Desktop/Soft_Robotics_Internship/YOLO_UR5_Test/yolov5/runs/train/exp8/weights/best.pt  --data data2.yaml