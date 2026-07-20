SEED = 42

# ----------------------------------- Dataset ----------------------------------- #
CLASSES = (
    "apple",
    "banana",
    "orange",
    "pineapple",
    "watermelon"
)

NUM_CLASSES = 5

TRAIN_CSV = "labels/train_labels.csv"
TRAIN_IMG_DIR = "imgs/dataset/train_imgs"

TEST_CSV = "labels/test_labels.csv"
TEST_IMG_DIR = "imgs/dataset/test_imgs"

# ----------------------------------- Training ----------------------------------- #

BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 10

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

# ------------------------------------ Model ------------------------------------ #

MODEL_NAME = "model_resnet18_transfer_learning"
MODEL_PATH = f"./models/{MODEL_NAME}.pth"