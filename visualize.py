import os
import torchvision.transforms.functional as TF


def write_image(item, root):
    if not os.path.exists(root):
        os.mkdir(root)
    for key in item:
        image = TF.to_pil_image(item[key])
        image.save(os.path.join(root, key))
