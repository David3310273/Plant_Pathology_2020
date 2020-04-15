import os
import torchvision.transforms.functional as TF
import csv


def write_image(item, root):
    if not os.path.exists(root):
        os.mkdir(root)
    for key in item:
        image = TF.to_pil_image(item[key])
        image.save(os.path.join(root, key))


# 结果输出到csv中用于比对
def write_csv(items, root, filename="test_result.csv"):
    path = os.path.join(root, filename)
    if not os.path.exists(root):
        os.mkdir(root)
    with open(path, "w", newline="") as f:
        header_writer = csv.DictWriter(f, fieldnames=["image_id", ""])
        header_writer.writeheader()
        writer = csv.writer(f)
        keys = sorted(items.keys(), key=lambda x: int(x.split("_")[1]))
        for key in keys:
            row = [key] + items[key].detach().cpu().numpy().tolist()
            writer.writerow(row)
    f.close()
