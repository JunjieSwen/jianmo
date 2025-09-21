#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
源数据RPM统计工具
"""

import os

def main():
    print("源数据RPM统计工具")
    print("=" * 40)
    
    # 数据路径
    data_path = 'data/'
    src_path = os.path.join(data_path, 'src')
    
    if not os.path.exists(src_path):
        print(f"错误：源数据路径不存在: {src_path}")
        return
    
    # 统计变量
    rpm_count = {}
    total_files = 0
    files_with_rpm = 0
    
    print("正在扫描源数据...")
    
    # 遍历所有数据目录
    for freq_dir in os.listdir(src_path):
        if not os.path.isdir(os.path.join(src_path, freq_dir)):
            continue
            
        freq_path = os.path.join(src_path, freq_dir)
        print(f"扫描目录: {freq_dir}")
        
        for fault_dir in os.listdir(fault_path):
            fault_path = os.path.join(freq_path, fault_dir)
            if not os.path.isdir(fault_path):
                continue
            
            # 扫描该目录下的所有.mat文件
            for root, dirs, files in os.walk(fault_path):
                for file in files:
                    if file.endswith('.mat'):
                        total_files += 1
                        
                        # 提取转速信息
                        rpm = None
                        if '1797rpm' in file:
                            rpm = 1797
                        elif '1772rpm' in file:
                            rpm = 1772
                        elif '1750rpm' in file:
                            rpm = 1750
                        elif '1730rpm' in file:
                            rpm = 1730
                        
                        if rpm is not None:
                            files_with_rpm += 1
                            if rpm not in rpm_count:
                                rpm_count[rpm] = 0
                            rpm_count[rpm] += 1
    
    print(f"扫描完成:")
    print(f"- 总文件数: {total_files}")
    print(f"- 有RPM信息的文件数: {files_with_rpm}")
    print(f"- 无RPM信息的文件数: {total_files - files_with_rpm}")
    
    print("\nRPM分布:")
    for rpm in sorted(rpm_count.keys()):
        print(f"RPM {rpm}: {rpm_count[rpm]} 个文件")
    
    print("\nRPM统计工具运行完成！")

if __name__ == "__main__":
    main()
