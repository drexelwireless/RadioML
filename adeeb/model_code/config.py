# some training parameters
EPOCHS = 10
BATCH_SIZE = 64
NUM_CLASSES = 11

image_height = 128
image_width = 128
channels = 3

save_model_dir = "../model"

dataset_dir = "data/train/"

train_img_dir = dataset_dir + "train_img.npy"
train_lbl_dir = dataset_dir + "train_lbl.npy"
# valid_dir = dataset_dir + "valid"
# test_dir = dataset_dir + "test"

# choose a network
model = "resnet18"
# model = "resnet34"
# model = "resnet50"
# model = "resnet101"
# model = "resnet152"
