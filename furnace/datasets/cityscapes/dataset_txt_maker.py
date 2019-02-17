"""
construct the txt files for cityscape dataset
"""
import os

root = "/datasets-ssd/cityscapes/"
fine_label_path = root + "gtFine/"
img_path = root + "leftImg8bit_trainvaltest/"

folders = ["train", "val", "test"]


def make_txt(type="label"):
    for mode in folders:
        if type == "label":
            path = fine_label_path
        else:
            path = img_path

        img_list_all = []
        path = os.path.join(path, mode)
        classes_list = os.listdir(path)
        classes_list.sort()

        # print(classes_list)
        for classes in classes_list:
            classes_img_path = os.path.join(path, classes)
            img_list = os.listdir(classes_img_path)
            img_list.sort()
            img_list_all.append(img_list)
            # print(img_list)



# with open("txts/train.txt", 'w') as f:


if __name__ == '__main__':
    make_txt("label")
