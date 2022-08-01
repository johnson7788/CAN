#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/7/31 15:04
# @File  : latex_predict_api.py
# @Author: 
# @Desc  : 图片到latex的api
######################################################
#
######################################################

import re
import sys
import json
import hashlib
import os
import time
import logging
from pathlib import Path
import requests
import cv2
import torch
from numpy import random
import base64
from flask import Flask, request, jsonify, abort
from utils import load_config, load_checkpoint, compute_edit_distance
from models.infer_model import Inference
from dataset import Words

app = Flask(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger("Main")

class LatexModel(object):
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.word_path = 'datasets/IM2LATEX/words_dict.txt'
        self.config_file = f'config_IM2LATEX.yaml'
        assert os.path.exists(self.config_file), f"{self.config_file}不存在,请检查"
        self.params = load_config(self.config_file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.params['device'] = self.device
        self.words = Words(self.word_path)
        self.params['word_num'] = len(self.words)
        if 'use_label_mask' not in self.params:
            self.params['use_label_mask'] = False
        self.draw_map = False
        self.load_predict_model()
        self.line_right = 0
        self.upload_dir = 'runs/api/uploads/'
        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir)

    def load_predict_model(self):
        # Load model
        model = Inference(self.params, draw_map=self.draw_map)
        self.model = model.to(self.device)
        load_checkpoint(self.model, None, self.params['checkpoint'])
        self.model.eval()
        logger.info(f"预测模型加载完成")

    def cal_md5(self,content):
        """
        计算content字符串的md5
        :param content:
        :return:
        """
        # 使用encode
        result = hashlib.md5(content.encode())
        # 打印hash
        md5 = result.hexdigest()
        return md5
    def download(self, image_url, image_path_name, force_download=False):
        """
        根据提供的图片的image_url，下载图片保存到image_path_name
        :param: force_download: 是否强制下载
        :return_exists: 是否图片已经存在，如果已经存在，那么返回2个True
        """
        # 如果存在图片，并且不强制下载，那么直接返回
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.132 Safari/537.36"
        }
        if os.path.exists(image_path_name) and not force_download:
            print(f"{image_path_name}图片已存在，不需要下载")
            return True
        try:
            response = requests.get(image_url, stream=True, headers=headers)
            with open(image_path_name, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            return True
        except Exception as e:
            print(f"{image_url}下载失败")
            print(f"{e}")
            return False
    def predict(self, image_name):
        with torch.no_grad():
            image_array = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            img = torch.Tensor(255 - image_array) / 255
            img = img.unsqueeze(0).unsqueeze(0)
            img = img.to(self.device)
            start_time = time.time()
            probs, _, = self.model.predict(img)
            model_time = time.time() - start_time
            # 解码成文字
            prediction = self.words.decode(probs)
            print(f"图片的是: {image_name}， 耗时是: {model_time}")
            print(f"预测的结果是: {prediction}, ")
        return prediction
    def predict_image(self, image_path):
        """
        返回的bboxes是实际的坐标，x1，y1，x2，y2，是左上角和右下角的坐标
        :param data: 图片数据的列表 [image1, image2]
        :return: [[images, bboxes, confidences, labels],[images, bboxes,confidences, labels],...] bboxes是所有的bboxes, confidence是置信度， labels是所有的bboxes对应的label，
        """
        #检查图片确实存在
        logger.info(f"预测本地的一个单张的图片")
        if os.path.exists(image_path):
            prediction = self.predict(image_path)
            results = {"code": 200, "msg": "success", "data": prediction}
        else:
            results = {"code": 400, "msg": "给定的图片不存在", "data": ""}
        return results
    def predict_url(self, url):
        """
        :param data: 一个图片的url，下载后预测
        :return: [[images, bboxes, confidences, labels],[images, bboxes,confidences, labels],...] bboxes是所有的bboxes, confidence是置信度， labels是所有的bboxes对应的label，
        """
        #检查图片确实存在
        logger.info(f"预测本地的一个的图片的url地址")
        image_suffix = url.split('.')[-1]
        md5 = self.cal_md5(url)
        new_image_name = md5 + '.' + image_suffix
        image_path = os.path.join(self.upload_dir, new_image_name)
        download_reuslt = self.download(image_url=url, image_path_name=image_path)
        if download_reuslt:
            prediction = self.predict(image_path)
            results = {"code": 200, "msg": "success", "data": prediction}
        else:
            results = {"code": 400, "msg": "图片下载失败，请检查图片地址是否正确", "data": ""}
        return results
    def predict_raw_image(self, raw_file, image_name=None):
        """
        :param raw_file: 图片的base64 str 格式
        :return:
        """
        #检查图片确实存在
        logger.info(f"预测本地的一个base64 str的图片")
        raw_str = raw_file.encode('utf-8')
        raw_img = base64.b64decode(raw_str)
        if image_name:
            img_name = os.path.join(self.upload_dir, image_name)
        else:
            img_name = os.path.join(self.upload_dir, 'tmp.jpg')
        with open(img_name, 'wb') as f:
            f.write(raw_img)
        prediction = self.predict(img_name)
        results = {"code": 200, "msg": "success", "data": prediction}
        return results

    def predict_directory(self,directory_path):
        """
        预测的是一个目的目录下的所有图片结果
        """
        logger.info(f"预测一个本地的目录，目录包含多张图片")
        if os.path.exists(directory_path):
            data = {}
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if file.endswith(".jpg") or file.endswith(".png"):
                        image_path = os.path.join(root, file)
                        prediction = self.predict(image_path)
                        data[file] = prediction
            results = {"code": 200, "msg": "success", "data": data}
        else:
            results = {"code": 400, "msg": "给定的目录不存在", "data": ""}
        return results


@app.route("/api/predict", methods=['POST'])
def predict():
    """
    接收POST请求，获取data参数,  bbox左上角的x1，y1, 右下角的x2,y2
    Args:
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    file_format = jsonres.get('format', "image")
    logger.info(f"收到的数据是:{test_data}")
    if file_format == "image":
        results = latex_model.predict_image(image_path=test_data)
    elif file_format == "directory":
        results = latex_model.predict_directory(directory_path=test_data)
    elif file_format == "base64":
        image_name = jsonres.get('name')
        results = latex_model.predict_raw_image(raw_file=test_data, image_name=image_name)
    elif file_format == "url":
        results = latex_model.predict_url(url=test_data)
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

if __name__ == "__main__":
    latex_model = LatexModel()
    app.run(host='0.0.0.0', port=7800, debug=False, threaded=True)

