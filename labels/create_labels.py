import os

label_file = "./labels/train_labels.csv"  # For Training Imgs
# label_file = "./labels/test_labels.csv"  # For Testing Imgs

img_dir = "./imgs/ref_imgs/train_imgs"  # For Training Imgs
# img_dir = "./imgs/ref_imgs/test_imgs"  # For Testing Imgs

imgs = os.listdir(img_dir)

lines = []

for img in imgs:
    img_name = img.split(".")[0][:-4]
    if img_name == "apple":
        classIdx = 0
    elif img_name == "banana":
        classIdx = 1
    elif img_name == "orange":
        classIdx = 2
    elif img_name == "pineapple":
        classIdx = 3
    elif img_name == "watermelon":
        classIdx = 4
    else:
        print(f"PEOS: {img_name}")

    lines.append(f"{img}, {classIdx}\n")

with open(label_file, "w") as f:
    f.writelines(lines)
