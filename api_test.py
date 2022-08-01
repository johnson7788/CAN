#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/7/31 19:16
# @File  : api_test.py
# @Author: 
# @Desc  : 测试api接口
import unittest
import requests
import time, os
import json
import base64

class ApiTestCase(unittest.TestCase):
    """

    """
    host = '127.0.0.1'
    def test_one_image(self):
        """
        测试一张图片，给定的是图片本地路径
        :return:
        """
        url = f"http://{self.host}:7800/api/predict"
        image_path = "datasets/test_images/76d30658bb.png"
        data = {"data": image_path, "format": "image"}
        start_time = time.time()
        # 提交form格式数据
        headers = {'content-type': 'application/json'}
        # 提交form格式数据
        r = requests.post(url, data=json.dumps(data), headers=headers)
        res = r.json()
        print(json.dumps(res, indent=4, ensure_ascii=False))
        print(f"花费时间: {time.time() - start_time}秒")
    def test_one_raw_image(self):
        """
        测试上传一张图片
        :return:
        """
        url = f"http://{self.host}:7800/api/predict"
        image_path = "datasets/test_images/76d30658bb.png"
        with open(image_path, 'rb') as f:
            img = base64.b64encode(f.read())
            img_str = img.decode('utf-8')
            data = {"data": img_str, "format": "base64", "name": "76d30658bb.png"}
        start_time = time.time()
        # 提交form格式数据
        headers = {'content-type': 'application/json'}
        # 提交form格式数据
        r = requests.post(url, data=json.dumps(data), headers=headers)
        res = r.json()
        print(json.dumps(res, indent=4, ensure_ascii=False))
        print(f"花费时间: {time.time() - start_time}秒")
    def test_one_url(self):
        """
        测试一张给定图片的url
        :return:
        """
        url = f"http://{self.host}:7800/api/predict"
        image_path = "https://pic1.zhimg.com/80/v2-87cc880e32fbe7b6ac426b3ab65115d8_1440w.jpg"
        data = {"data": image_path, "format": "url"}
        start_time = time.time()
        # 提交form格式数据
        headers = {'content-type': 'application/json'}
        # 提交form格式数据
        r = requests.post(url, data=json.dumps(data), headers=headers)
        res = r.json()
        print(json.dumps(res, indent=4, ensure_ascii=False))
        print(f"花费时间: {time.time() - start_time}秒")
    def test_image_dir(self):
        """
        测试一张图片，给定的是图片本地路径
        :return:
        """
        url = f"http://{self.host}:7800/api/predict"
        image_path = "datasets/test_images"
        data = {"data": image_path, "format": "directory"}
        start_time = time.time()
        # 提交form格式数据
        headers = {'content-type': 'application/json'}
        # 提交form格式数据
        r = requests.post(url, data=json.dumps(data), headers=headers)
        res = r.json()
        print(json.dumps(res, indent=4, ensure_ascii=False))
        print(f"花费时间: {time.time() - start_time}秒")

if __name__ == '__main__':
    print()