#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
源数据RPM统计工具
统计源数据中所有文件的RPM信息并保存为txt文件
"""

import os
from collections import defaultdict, Counter
from datetime import datetime

def extract_rpm_from_mat_file(file_path):
    """从.mat文件中提取转速"""
    try:
        from scipy.io import loadmat
        mat_data = loadmat(file_path)
        
        # 查找RPM变量
        if 'RPM' in mat_data:
            rpm_value = mat_data['RPM']
            # 确保RPM值是标量
            if hasattr(rpm_value, 'item'):
                return rpm_value.item()
            elif isinstance(rpm_value, (int, float)):
                return float(rpm_value)
            else:
                # 如果是数组，取第一个值
                return float(rpm_value.flatten()[0])
        else:
            return None  # 没有RPM变量
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None

def determine_fault_type(fault_dir):
    """确定故障类型"""
    if fault_dir.startswith('B'):
        return 'BS'  # 滚动体故障
    elif fault_dir.startswith('IR'):
        return 'IR'  # 内圈故障
    elif fault_dir.startswith('OR'):
        return 'OR'  # 外圈故障
    elif fault_dir == 'N':
        return 'N'   # 正常状态
    else:
        return 'Unknown'

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
    rpm_stats = defaultdict(list)
    fault_type_stats = defaultdict(lambda: defaultdict(int))
    freq_stats = defaultdict(lambda: defaultdict(int))
    all_files = []
    
    total_files = 0
    files_with_rpm = 0
    
    print("正在扫描源数据...")
    
    # 遍历所有数据目录
    for freq_dir in os.listdir(src_path):
        if not os.path.isdir(os.path.join(src_path, freq_dir)):
            continue
            
        freq_path = os.path.join(src_path, freq_dir)
        print(f"扫描目录: {freq_dir}")
        
        for fault_dir in os.listdir(freq_path):
            fault_path = os.path.join(freq_path, fault_dir)
            if not os.path.isdir(fault_path):
                continue
            
            # 确定故障类型
            fault_type = determine_fault_type(fault_dir)
            
            # 扫描该目录下的所有.mat文件
            for root, dirs, files in os.walk(fault_path):
                for file in files:
                    if file.endswith('.mat'):
                        total_files += 1
                        file_path = os.path.join(root, file)
                        
                        # 提取转速信息
                        rpm = extract_rpm_from_mat_file(file_path)
                        
                        # 记录文件信息
                        file_info = {
                            'file_path': file_path,
                            'filename': file,
                            'fault_type': fault_type,
                            'freq_dir': freq_dir,
                            'fault_dir': fault_dir,
                            'rpm': rpm
                        }
                        
                        all_files.append(file_info)
                        
                        if rpm is not None:
                            files_with_rpm += 1
                            rpm_stats[rpm].append(file_info)
                            fault_type_stats[fault_type][rpm] += 1
                            freq_stats[freq_dir][rpm] += 1
    
    print(f"扫描完成:")
    print(f"- 总文件数: {total_files}")
    print(f"- 有RPM信息的文件数: {files_with_rpm}")
    print(f"- 无RPM信息的文件数: {total_files - files_with_rpm}")
    
    # 生成报告
    print("正在生成统计报告...")
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("源数据RPM统计报告")
    report_lines.append("=" * 60)
    report_lines.append(f"生成时间: {current_time}")
    report_lines.append(f"数据路径: {data_path}")
    report_lines.append("")
    
    # 1. 总体统计
    report_lines.append("1. 总体统计")
    report_lines.append("-" * 30)
    files_without_rpm = total_files - files_with_rpm
    
    report_lines.append(f"总文件数: {total_files}")
    report_lines.append(f"有RPM信息的文件数: {files_with_rpm}")
    report_lines.append(f"无RPM信息的文件数: {files_without_rpm}")
    report_lines.append(f"RPM覆盖率: {files_with_rpm/total_files*100:.1f}%")
    report_lines.append("")
    
    # 2. RPM分布统计
    report_lines.append("2. RPM分布统计")
    report_lines.append("-" * 30)
    rpm_counter = Counter()
    for rpm, files in rpm_stats.items():
        rpm_counter[rpm] = len(files)
    
    for rpm in sorted(rpm_counter.keys()):
        count = rpm_counter[rpm]
        percentage = count / files_with_rpm * 100
        report_lines.append(f"RPM {rpm}: {count} 个文件 ({percentage:.1f}%)")
    report_lines.append("")
    
    # 3. 故障类型RPM分布
    report_lines.append("3. 故障类型RPM分布")
    report_lines.append("-" * 30)
    for fault_type in sorted(fault_type_stats.keys()):
        report_lines.append(f"{fault_type}故障:")
        fault_rpm_stats = fault_type_stats[fault_type]
        total_fault_files = sum(fault_rpm_stats.values())
        
        for rpm in sorted(fault_rpm_stats.keys()):
            count = fault_rpm_stats[rpm]
            percentage = count / total_fault_files * 100
            report_lines.append(f"  RPM {rpm}: {count} 个文件 ({percentage:.1f}%)")
        report_lines.append(f"  总计: {total_fault_files} 个文件")
        report_lines.append("")
    
    # 4. 采样频率RPM分布
    report_lines.append("4. 采样频率RPM分布")
    report_lines.append("-" * 30)
    for freq_dir in sorted(freq_stats.keys()):
        report_lines.append(f"{freq_dir}:")
        freq_rpm_stats = freq_stats[freq_dir]
        total_freq_files = sum(freq_rpm_stats.values())
        
        for rpm in sorted(freq_rpm_stats.keys()):
            count = freq_rpm_stats[rpm]
            percentage = count / total_freq_files * 100
            report_lines.append(f"  RPM {rpm}: {count} 个文件 ({percentage:.1f}%)")
        report_lines.append(f"  总计: {total_freq_files} 个文件")
        report_lines.append("")
    
    # 5. 详细文件列表
    report_lines.append("5. 详细文件列表")
    report_lines.append("-" * 30)
    
    for rpm in sorted(rpm_stats.keys()):
        files = rpm_stats[rpm]
        report_lines.append(f"RPM {rpm} 文件列表 ({len(files)} 个文件):")
        
        for file_info in sorted(files, key=lambda x: x['file_path']):
            relative_path = os.path.relpath(file_info['file_path'], data_path)
            report_lines.append(f"  {relative_path}")
        report_lines.append("")
    
    # 6. 无RPM信息的文件列表
    files_without_rpm_list = [f for f in all_files if f['rpm'] is None]
    if files_without_rpm_list:
        report_lines.append("6. 无RPM信息的文件列表")
        report_lines.append("-" * 30)
        report_lines.append(f"共 {len(files_without_rpm_list)} 个文件:")
        
        for file_info in sorted(files_without_rpm_list, key=lambda x: x['file_path']):
            relative_path = os.path.relpath(file_info['file_path'], data_path)
            report_lines.append(f"  {relative_path}")
        report_lines.append("")
    
    # 7. 统计摘要
    report_lines.append("7. 统计摘要")
    report_lines.append("-" * 30)
    report_lines.append(f"发现的RPM值: {sorted(rpm_counter.keys())}")
    report_lines.append(f"故障类型: {sorted(fault_type_stats.keys())}")
    report_lines.append(f"采样频率: {sorted(freq_stats.keys())}")
    report_lines.append("")
    
    report_lines.append("=" * 60)
    report_lines.append("报告结束")
    report_lines.append("=" * 60)
    
    # 保存报告
    output_file = 'rpm_statistics.txt'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in report_lines:
                f.write(line + '\n')
        print(f"统计报告已保存到: {output_file}")
        
        # 打印简要统计信息
        print("\n简要统计信息:")
        print("-" * 20)
        print(f"总文件数: {total_files}")
        print(f"有RPM信息的文件数: {files_with_rpm}")
        print(f"无RPM信息的文件数: {total_files - files_with_rpm}")
        
        if rpm_stats:
            print(f"发现的RPM值: {sorted(rpm_stats.keys())}")
            for rpm in sorted(rpm_stats.keys()):
                print(f"RPM {rpm}: {len(rpm_stats[rpm])} 个文件")
        
        print("\nRPM统计工具运行成功！")
        
    except Exception as e:
        print(f"保存报告时出错: {e}")

if __name__ == "__main__":
    main()
