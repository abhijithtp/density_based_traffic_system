
!pip install ultralytics
from ultralytics import YOLO
import re
import pandas
# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on an image ,
#results = model('/content/cars.jpg')  # results list
results=model.predict('/content/cars1.jpg', save=True, imgsz=320, conf=0.4,classes=[2,3,5,7],show_labels=True,show_boxes=True,show_conf=False)
#imgsz=320 can vary it

#to get the count of vehicles as a varible 
counts = {}
for result in results:
    boxes = result.boxes.cpu().numpy()
    for box in boxes:
        cls = int(box.cls[0])
        if not cls in counts.keys():
            counts[cls] = 1
        else:
            counts[cls] += 1
for key in counts.keys():
    print(model.names[key] + " : " + str(counts[key]))
#weights of vehicles
class_multipliers = {
    2: 1,  # Multiplier for class 2
    3: 1,  # Multiplier for class 3
    5: 2,  # Multiplier for class 5
    7: 4   # Multiplier for class 7
}
sum=0
n=0
for i in [2,3,5,7]:
  try:
    sum+=counts[i]*class_multipliers[i]
    n+=counts[i]
  except:
    pass
avg=sum/n
print('Time alloted:'+str((n*27)/avg))
