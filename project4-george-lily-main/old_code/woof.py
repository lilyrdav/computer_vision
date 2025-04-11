from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Would you like to use pretrained weights?
SKIP_TRAINING = True

# Load pretrained YOLOv8 nano (n) classification (cls) model
# https://docs.ultralytics.com/datasets/classify/mnist/
model = YOLO('woof-yolov8n-cls.pt') if SKIP_TRAINING else YOLO('yolov8n-cls.pt')

# Grab MNIST and train the model
results = model.val(data='imagewoof320', imgsz=224) if SKIP_TRAINING else model.train(data='mnist', epochs=5, imgsz=224)

# Read images
# mode = "val" if SKIP_TRAINING else "train"
# img_label = mpimg.imread('./runs/classify/' + mode + '/val_batch0_labels.jpg')
# img_pred = mpimg.imread('./runs/classify/' + mode + '/val_batch0_pred.jpg')

# # display images
# fig, ax = plt.subplots(1,2)
# ax[0].set_axis_off()
# ax[0].imshow(img_label)
# ax[1].set_axis_off()
# ax[1].imshow(img_pred)