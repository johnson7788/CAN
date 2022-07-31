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
    def test_one_url(self):
        """
        测试一张给定图片的url
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