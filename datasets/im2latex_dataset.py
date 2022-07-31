#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/7/30 09:48
# @File  : im2latex_dataset.py.py
# @Author: 
# @Desc  : 读取im2latex数据集,图片转成pkl格式，标签转成txt格式

import os
import json
import pickle
import cv2
from tqdm import tqdm

def read_im2latex_dataset(source_path="/Users/admin/git/image-to-latex/data", save_path="IM2LATEX", skip_image=False):
    """
    读取im2latex数据集,图片转成pkl格式，标签转成txt格式
    :param source_path: 数据集是经过image-to-latex进行预处理过的
    :param save_path:
    :return:
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    formula_images = os.path.join(source_path, "formula_images_processed")
    formula_text = os.path.join(source_path, "im2latex_formulas.norm.new.lst")
    train_text = os.path.join(source_path, "im2latex_train_filter.lst")
    test_text = os.path.join(source_path, "im2latex_test_filter.lst")
    val_text = os.path.join(source_path, "im2latex_validate_filter.lst")
    src_vocab = os.path.join(source_path, "vocab.json")  # 读取已有vocab或自己做一个
    # 读取所有的latex
    formula_text_dict = {}
    with open(formula_text, "r") as f:
        idx = 0
        for line in f:
            line = line.strip()
            formula_text_dict[idx] = line
            idx += 1
    if not skip_image:
        # 读取所有图片,图片名字作为key,图片内容的numpy格式作为value
        formula_images_dict = {}
        formula_images_list = os.listdir(formula_images)
        for formula_image in tqdm(formula_images_list, desc="读取图片中"):
            formula_image_path = os.path.join(formula_images, formula_image)
            formula_image_content = cv2.imread(formula_image_path, cv2.IMREAD_GRAYSCALE)
            formula_images_dict[formula_image] = formula_image_content
        # 读取训练集, 并保存成新的训练集
        def get_image_text_and_save(handle_data, save_name="train"):
            get_image = {}
            get_text = {}
            with open(handle_data, "r") as f:
                for line in f:
                    line = line.strip()
                    line_split = line.split(" ")
                    image_name = line_split[0]
                    formula_idx = int(line_split[1])
                    if image_name in formula_images_dict and formula_idx in formula_text_dict:
                        get_text[image_name] = formula_text_dict[formula_idx]
                        get_image[image_name] = formula_images_dict[image_name]
            labels = os.path.join(save_path, save_name + "_labels.txt")
            images = os.path.join(save_path, save_name + "_images.pkl")
            with open(labels, "w") as f:
                for name, form in get_text.items():
                    f.write(name + "\t" + form + "\n")
            with open(images, "wb") as f:
                pickle.dump(get_image, f)
            print(f"保存标签text文件{labels}")
            print(f"保存图片pkl文件{images}")
            return get_image, get_text
        train_image, train_text = get_image_text_and_save(train_text, save_name="train")
        test_image, test_text = get_image_text_and_save(test_text, save_name="test")
        val_image, val_text = get_image_text_and_save(val_text, save_name="val")
    # 对vocab字典进行处理
    vocab = ["eos","sos"]
    #如果特殊字符在not_vocab，那么需要剔除
    not_vocab = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
    # with open(src_vocab, "r") as f:
    #     json_data = json.load(f)
    # for key, _ in json_data.items():
    #     vocab.append(key)
    # 手动生成字典
    for item in formula_text_dict.values():
        item_list = item.strip().split()
        for word in item_list:
            if word not in vocab:
                vocab.append(word)
    print(f"vocab的长度是{len(vocab)}")
    # 保存到文件
    vocab_path = os.path.join(save_path, "words_dict.txt")
    with open(vocab_path, "w") as f:
        for word in vocab:
            if word not in not_vocab:
                f.write(word + "\n")
    print(f"数据集处理完成")

if __name__ == '__main__':
    read_im2latex_dataset(skip_image=True)