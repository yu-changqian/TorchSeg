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

        path = os.path.join(path, mode)
        classes_list = os.listdir(path)
        classes_list.sort()

        for classes in classes_list:
            classes_img_path = os.path.join(path, classes)
            img_list = os.listdir(classes_img_path)
            img_list.sort()

            txt_path = "txts/" + mode + ".txt"
            with open(txt_path, 'a') as f:
                for imgs in img_list:
                    # print(imgs)
                    name = imgs.split("_")
                    # print(name)
                    if name[4] == "color.png":
                        label_name = imgs
                        # print(imgs)
                        img_name = name[0] + "_" + name[1] + "_" + name[2] + "_leftImg8bit.png"
                        # print(img_name)
                        path_to_label = classes_img_path + "/" + label_name
                        path_to_img = img_path + mode + "/" + classes + "/" + img_name

                        # print(path_to_img)
                        # print(path_to_label)
                        # exit()
                        line = path_to_img + "   " + path_to_label
                        f.writelines(line+"\n")


if __name__ == '__main__':
    make_txt("label")
