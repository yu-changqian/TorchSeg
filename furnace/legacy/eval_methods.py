import numpy as np
import cv2
import torch
from torch.autograd import Variable
from torch.utils import data

from utils.img_utils import pad_image_to_shape, normalize
from utils.pyt_utils import load_model


def whole_eval(val_func, class_num, img, img_means, img_std, scale_array,
               output_size, is_flip, device=None):
    # ori_h, ori_w, c = img.shape
    processed_pred = np.zeros((output_size[0], output_size[1], class_num))

    for s in scale_array:
        scaled_img = cv2.resize(img, None, fx=s, fy=s,
                                interpolation=cv2.INTER_LINEAR)
        scaled_img = pre_img(scaled_img, img_means, img_std, None)
        pred = val_func_process(val_func, scaled_img, is_flip, device)
        pred = pred.permute(1, 2, 0)
        processed_pred += cv2.resize(pred.cpu().numpy(),
                                     (output_size[1], output_size[0]),
                                     interpolation=cv2.INTER_LINEAR)

    pred = processed_pred.argmax(2)

    return pred


def sliding_eval(val_func, class_num, img, img_means, img_std,
                 crop_size, scale_array, is_flip, device=None):
    ori_rows, ori_cols, c = img.shape
    data_all = np.zeros((ori_rows, ori_cols, class_num))

    for s in scale_array:
        img_scale = cv2.resize(img, None, fx=s, fy=s,
                               interpolation=cv2.INTER_LINEAR)
        new_rows, new_cols, _ = img_scale.shape
        data_all += scale_process(val_func, class_num, img_scale,
                                  (ori_rows, ori_cols), img_means, img_std,
                                  crop_size, is_flip, device)

    pred = data_all.argmax(2)

    return pred


def scale_process(val_func, class_num, img_scale, ori_shape, img_means, img_std,
                  crop_size, is_flip=False, device=None):
    new_rows, new_cols, c = img_scale.shape
    long_size = new_cols if new_cols > new_rows else new_rows

    if long_size <= crop_size:
        input_data, margin = pre_img(img_scale, img_means, img_std, crop_size)
        score = val_func_process(val_func, input_data, is_flip, device)
        score = score[:, margin[0]:(score.shape[1] - margin[1]),
                margin[2]:(score.shape[2] - margin[3])]
    else:
        stride_rate = 2 / 3
        stride = int(np.ceil(crop_size * stride_rate))
        # stride = crop_size - 170
        img_pad = img_scale

        img_pad, margin = pad_image_to_shape(img_pad, crop_size,
                                             cv2.BORDER_CONSTANT, value=0)

        pad_rows = img_pad.shape[0]
        pad_cols = img_pad.shape[1]
        r_grid = int(np.ceil((pad_rows - crop_size) / stride)) + 1
        c_grid = int(np.ceil((pad_cols - crop_size) / stride)) + 1
        data_scale = torch.zeros(class_num, pad_rows, pad_cols).cuda(device)
        count_scale = torch.zeros(class_num, pad_rows, pad_cols).cuda(device)

        for grid_yidx in range(r_grid):
            for grid_xidx in range(c_grid):
                s_x = grid_xidx * stride
                s_y = grid_yidx * stride
                e_x = min(s_x + crop_size, pad_cols)
                e_y = min(s_y + crop_size, pad_rows)
                s_x = e_x - crop_size
                s_y = e_y - crop_size
                img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                count_scale[:, s_y: e_y, s_x: e_x] += 1

                input_data, tmargin = pre_img(img_sub, img_means,
                                              img_std, crop_size)
                temp_score = val_func_process(val_func, input_data, is_flip,
                                              device)
                temp_score = temp_score[:,
                             tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                             tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                data_scale[:, s_y: e_y, s_x: e_x] += temp_score
        # score = data_scale / count_scale
        score = data_scale
        score = score[:, margin[0]:(score.shape[1] - margin[1]),
                margin[2]:(score.shape[2] - margin[3])]

    score = score.permute(1, 2, 0)
    ori_shape = (ori_shape[1], ori_shape[0])
    data_output = cv2.resize(score.cpu().numpy(), ori_shape,
                             interpolation=cv2.INTER_LINEAR)

    return data_output


def val_func_process(val_func, input_data, is_flip=False, device=None):
    input_data = np.ascontiguousarray(input_data[None, :, :, :],
                                      dtype=np.float32)
    input_data = torch.FloatTensor(input_data).cuda(device)

    with torch.cuda.device(input_data.get_device()):
        val_func.eval()
        val_func.to(input_data.get_device())
        with torch.no_grad():
            score = val_func(input_data)
            score = score[0]

            if is_flip:
                input_data = input_data.flip(-1)
                score_flip = val_func(input_data)
                score_flip = score_flip[0]
                score += score_flip.flip(-1)
            score = torch.exp(score)
            score = score.data

    return score


def pre_img(img, image_means, image_std, crop_size=None):
    p_img = img

    if img.shape[2] < 3:
        im_b = p_img
        im_g = p_img
        im_r = p_img
        p_img = np.concatenate((im_b, im_g, im_r), axis=2)

    p_img = normalize(p_img, image_means, image_std)

    if crop_size is not None:
        p_img, margin = pad_image_to_shape(p_img, crop_size,
                                           cv2.BORDER_CONSTANT, value=0)
        p_img = p_img.transpose(2, 0, 1)

        return p_img, margin

    p_img = p_img.transpose(2, 0, 1)

    return p_img
