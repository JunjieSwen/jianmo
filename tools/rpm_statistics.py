#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
源数据RPM统计工具
统计源数据中所有文件的RPM信息并保存为txt文件
"""

import os
import sys
from collections import defaultdict, Counter
from datetime import datetime

class RPMStatisticsTool:
    """RPM统计工具"""
    
    def __init__(self, data_path='../data/'):
        self.data_path = data_path
        self.rpm_stats = defaultdict(list)  # RPM -> 文件列表
        self.fault_type_stats = defaultdict(lambda: defaultdict(int))  # 故障类型 -> RPM -> 数量
        self.freq_stats = defaultdict(lambda: defaultdict(int))  # 采样频率 -> RPM -> 数量
        self.all_files = []  # 所有文件信息
        
    def extract_rpm_from_filename(self, filename):
        """从文件名提取转速"""
        if '1797rpm' in filename:
            return 1797
        elif '1772rpm' in filename:
            return 1772
        elif '1750rpm' in filename:
            return 1750
        elif '1730rpm' in filename:
            return 1730
        else:
            return None  # 没有转速信息
    
    def determine_fault_type(self, fault_dir):
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
    
    def scan_source_data(self):
        """扫描源数据"""
        print("正在扫描源数据...")
        
        src_path = os.path.join(self.data_path, 'src')
        if not os.path.exists(src_path):
            print(f"错误：源数据路径不存在: {src_path}")
            return False
        
        total_files = 0
        files_with_rpm = 0
        
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
                fault_type = self.determine_fault_type(fault_dir)
                
                # 扫描该目录下的所有.mat文件
                for root, dirs, files in os.walk(fault_path):
                    for file in files:
                        if file.endswith('.mat'):
                            total_files += 1
                            file_path = os.path.join(root, file)
                            
                            # 提取转速信息
                            rpm = self.extract_rpm_from_filename(file)
                            
                            # 记录文件信息
                            file_info = {
                                'file_path': file_path,
                                'filename': file,
                                'fault_type': fault_type,
                                'freq_dir': freq_dir,
                                'fault_dir': fault_dir,
                                'rpm': rpm
                            }
                            
                            self.all_files.append(file_info)
                            
                            if rpm is not None:
                                files_with_rpm += 1
                                # 统计RPM分布
                                self.rpm_stats[rpm].append(file_info)
                                self.fault_type_stats[fault_type][rpm] += 1
                                self.freq_stats[freq_dir][rpm] += 1
        
        print(f"扫描完成:")
        print(f"- 总文件数: {total_files}")
        print(f"- 有RPM信息的文件数: {files_with_rpm}")
        print(f"- 无RPM信息的文件数: {total_files - files_with_rpm}")
        
        return True
    
    def generate_statistics_report(self):
        """生成统计报告"""
        print("正在生成统计报告...")
        
        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 创建报告内容
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("源数据RPM统计报告")
        report_lines.append("=" * 60)
        report_lines.append(f"生成时间: {current_time}")
        report_lines.append(f"数据路径: {self.data_path}")
        report_lines.append("")
        
        # 1. 总体统计
        report_lines.append("1. 总体统计")
        report_lines.append("-" * 30)
        total_files = len(self.all_files)
        files_with_rpm = sum(len(files) for files in self.rpm_stats.values())
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
        for rpm, files in self.rpm_stats.items():
            rpm_counter[rpm] = len(files)
        
        for rpm in sorted(rpm_counter.keys()):
            count = rpm_counter[rpm]
            percentage = count / files_with_rpm * 100
            report_lines.append(f"RPM {rpm}: {count} 个文件 ({percentage:.1f}%)")
        report_lines.append("")
        
        # 3. 故障类型RPM分布
        report_lines.append("3. 故障类型RPM分布")
        report_lines.append("-" * 30)
        for fault_type in sorted(self.fault_type_stats.keys()):
            report_lines.append(f"{fault_type}故障:")
            fault_rpm_stats = self.fault_type_stats[fault_type]
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
        for freq_dir in sorted(self.freq_stats.keys()):
            report_lines.append(f"{freq_dir}:")
            freq_rpm_stats = self.freq_stats[freq_dir]
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
        
        # 按RPM分组显示文件
        for rpm in sorted(self.rpm_stats.keys()):
            files = self.rpm_stats[rpm]
            report_lines.append(f"RPM {rpm} 文件列表 ({len(files)} 个文件):")
            
            for file_info in sorted(files, key=lambda x: x['file_path']):
                relative_path = os.path.relpath(file_info['file_path'], self.data_path)
                report_lines.append(f"  {relative_path}")
            report_lines.append("")
        
        # 6. 无RPM信息的文件列表
        files_without_rpm_list = [f for f in self.all_files if f['rpm'] is None]
        if files_without_rpm_list:
            report_lines.append("6. 无RPM信息的文件列表")
            report_lines.append("-" * 30)
            report_lines.append(f"共 {len(files_without_rpm_list)} 个文件:")
            
            for file_info in sorted(files_without_rpm_list, key=lambda x: x['file_path']):
                relative_path = os.path.relpath(file_info['file_path'], self.data_path)
                report_lines.append(f"  {relative_path}")
            report_lines.append("")
        
        # 7. 统计摘要
        report_lines.append("7. 统计摘要")
        report_lines.append("-" * 30)
        report_lines.append(f"发现的RPM值: {sorted(rpm_counter.keys())}")
        report_lines.append(f"故障类型: {sorted(self.fault_type_stats.keys())}")
        report_lines.append(f"采样频率: {sorted(self.freq_stats.keys())}")
        report_lines.append("")
        
        report_lines.append("=" * 60)
        report_lines.append("报告结束")
        report_lines.append("=" * 60)
        
        return report_lines
    
    def save_report_to_file(self, report_lines, output_file='rpm_statistics.txt'):
        """保存报告到文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in report_lines:
                    f.write(line + '\n')
            print(f"统计报告已保存到: {output_file}")
            return True
        except Exception as e:
            print(f"保存报告时出错: {e}")
            return False
    
    def run_statistics(self):
        """运行统计"""
        print("源数据RPM统计工具")
        print("=" * 40)
        
        # 扫描源数据
        if not self.scan_source_data():
            return False
        
        # 生成统计报告
        report_lines = self.generate_statistics_report()
        
        # 保存报告
        output_file = 'rpm_statistics.txt'
        if self.save_report_to_file(report_lines, output_file):
            print(f"\n统计完成！报告已保存为: {output_file}")
            
            # 打印简要统计信息
            print("\n简要统计信息:")
            print("-" * 20)
            total_files = len(self.all_files)
            files_with_rpm = sum(len(files) for files in self.rpm_stats.values())
            print(f"总文件数: {total_files}")
            print(f"有RPM信息的文件数: {files_with_rpm}")
            print(f"无RPM信息的文件数: {total_files - files_with_rpm}")
            
            if self.rpm_stats:
                print(f"发现的RPM值: {sorted(self.rpm_stats.keys())}")
                for rpm in sorted(self.rpm_stats.keys()):
                    print(f"RPM {rpm}: {len(self.rpm_stats[rpm])} 个文件")
            
            return True
        else:
            return False

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = '../data/'
    
    # 创建统计工具
    tool = RPMStatisticsTool(data_path)
    
    # 运行统计
    success = tool.run_statistics()
    
    if success:
        print("\nRPM统计工具运行成功！")
    else:
        print("\nRPM统计工具运行失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
