#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
源数据RPM统计工具（测试版）
"""

import os

def main():
    print("RPM统计工具测试")
    print("=" * 30)
    
    # 检查数据路径
    data_path = 'data/'
    src_path = os.path.join(data_path, 'src')
    
    print(f"数据路径: {data_path}")
    print(f"源数据路径: {src_path}")
    print(f"源数据路径是否存在: {os.path.exists(src_path)}")
    
    if os.path.exists(src_path):
        print("源数据目录内容:")
        for item in os.listdir(src_path):
            print(f"  {item}")
    else:
        print("源数据路径不存在！")
        return
    
    # 测试文件名解析
    test_filenames = [
        "B007_0.mat",
        "B028_0_(1797rpm).mat",
        "B028_1_(1772rpm).mat",
        "B028_2_(1750rpm).mat",
        "B028_3_(1730rpm).mat",
        "IR007_0.mat",
        "IR028_0_(1797rpm).mat",
        "N_0.mat",
        "N_1_(1772rpm).mat"
    ]
    
    print("\n测试文件名解析:")
    for filename in test_filenames:
        rpm = None
        if '1797rpm' in filename:
            rpm = 1797
        elif '1772rpm' in filename:
            rpm = 1772
        elif '1750rpm' in filename:
            rpm = 1750
        elif '1730rpm' in filename:
            rpm = 1730
        
        print(f"  {filename} -> RPM: {rpm}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()
