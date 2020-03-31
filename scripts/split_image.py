import os
import shutil

def split_images(root):
    img_path = os.path.join(root, "images")
    images = os.listdir(img_path)
    train_target = os.path.join(root, "train")
    test_target = os.path.join(root, "test")
    for image in images:
        if image.startswith("Train"):
            shutil.move(os.path.join(img_path, image), train_target)
            print("Moving {} to {}".format(image, train_target))
        elif image.startswith("Test"):
            shutil.move(os.path.join(img_path, image), test_target)
            print("Moving {} to {}".format(image, test_target))

if __name__ == '__main__':
    split_images("/Users/david/Desktop/plant-pathology-2020-fgvc7")

