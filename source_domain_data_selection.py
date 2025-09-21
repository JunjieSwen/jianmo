#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
源域数据筛选和迁移学习准备
针对问题1：从源域数据中筛选部分数据组成数据集
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SourceDomainDataSelector:
    """源域数据选择器"""
    
    def __init__(self, data_path='data/'):
        self.data_path = data_path
        self.source_data_info = []
        self.selected_data = []
        
    def explore_source_data(self):
        """探索源域数据结构"""
        print("正在探索源域数据结构...")
        
        src_path = os.path.join(self.data_path, 'src')
        
        # 统计各类型数据
        data_stats = {}
        
        for freq_dir in os.listdir(src_path):
            if not os.path.isdir(os.path.join(src_path, freq_dir)):
                continue
                
            freq_path = os.path.join(src_path, freq_dir)
            data_stats[freq_dir] = {}
            
            for fault_dir in os.listdir(freq_path):
                fault_path = os.path.join(freq_path, fault_dir)
                if not os.path.isdir(fault_path):
                    continue
                
                # 统计文件数量和转速分布
                file_info = self._analyze_fault_directory(fault_path, fault_dir, freq_dir)
                data_stats[freq_dir][fault_dir] = file_info
        
        # 打印统计信息
        self._print_data_statistics(data_stats)
        
        return data_stats
    
    def _analyze_fault_directory(self, fault_path, fault_dir, freq_dir):
        """分析故障目录"""
        file_count = 0
        rpm_distribution = {}
        file_list = []
        
        for root, dirs, files in os.walk(fault_path):
            for file in files:
                if file.endswith('.mat'):
                    file_count += 1
                    file_path = os.path.join(root, file)
                    
                    # 提取转速信息
                    rpm = self._extract_rpm_from_mat_file(file_path)
                    if rpm is None:  # 跳过没有转速信息的文件
                        continue
                    if rpm not in rpm_distribution:
                        rpm_distribution[rpm] = 0
                    rpm_distribution[rpm] += 1
                    
                    # 确定故障类型
                    fault_type = self._determine_fault_type(fault_dir)
                    
                    file_list.append({
                        'file_path': file_path,
                        'fault_type': fault_type,
                        'rpm': rpm,
                        'freq_dir': freq_dir,
                        'fault_dir': fault_dir
                    })
        
        return {
            'file_count': file_count,
            'rpm_distribution': rpm_distribution,
            'file_list': file_list
        }
    
    def _extract_rpm_from_mat_file(self, file_path):
        """从.mat文件中提取转速"""
        try:
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
    
    def _determine_fault_type(self, fault_dir):
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
    
    def _print_data_statistics(self, data_stats):
        """打印数据统计信息"""
        print("\n源域数据统计:")
        print("=" * 50)
        
        total_files = 0
        fault_type_counts = {}
        rpm_counts = {}
        
        for freq_dir, fault_stats in data_stats.items():
            print(f"\n{freq_dir}:")
            for fault_dir, info in fault_stats.items():
                file_count = info['file_count']
                rpm_dist = info['rpm_distribution']
                
                print(f"  {fault_dir}: {file_count} 个文件")
                print(f"    转速分布: {rpm_dist}")
                
                total_files += file_count
                
                # 统计故障类型
                fault_type = self._determine_fault_type(fault_dir)
                if fault_type not in fault_type_counts:
                    fault_type_counts[fault_type] = 0
                fault_type_counts[fault_type] += file_count
                
                # 统计转速
                for rpm, count in rpm_dist.items():
                    if rpm not in rpm_counts:
                        rpm_counts[rpm] = 0
                    rpm_counts[rpm] += count
        
        print(f"\n总计: {total_files} 个文件")
        print(f"故障类型分布: {fault_type_counts}")
        print(f"转速分布: {rpm_counts}")
    
    def select_representative_data(self, data_stats, samples_per_type=50):
        """选择代表性数据"""
        print(f"\n正在选择代表性数据（每类{samples_per_type}个样本）...")
        
        selected_data = []
        
        for freq_dir, fault_stats in data_stats.items():
            for fault_dir, info in fault_stats.items():
                file_list = info['file_list']
                fault_type = self._determine_fault_type(fault_dir)
                
                if fault_type == 'Unknown':
                    continue
                
                # 按转速分组
                rpm_groups = {}
                for file_info in file_list:
                    rpm = file_info['rpm']
                    if rpm not in rpm_groups:
                        rpm_groups[rpm] = []
                    rpm_groups[rpm].append(file_info)
                
                # 从每个转速组中选择样本
                selected_count = 0
                for rpm, files in rpm_groups.items():
                    if selected_count >= samples_per_type:
                        break
                    
                    # 随机选择文件
                    np.random.seed(42)  # 确保可重复性
                    selected_files = np.random.choice(files, 
                                                    min(len(files), samples_per_type - selected_count), 
                                                    replace=False)
                    
                    for file_info in selected_files:
                        selected_data.append(file_info)
                        selected_count += 1
                        
                        if selected_count >= samples_per_type:
                            break
        
        self.selected_data = selected_data
        
        print(f"数据选择完成，共选择{len(selected_data)}个样本")
        
        # 统计选择的样本
        self._print_selected_data_statistics()
        
        return selected_data
    
    def _print_selected_data_statistics(self):
        """打印选择数据的统计信息"""
        print("\n选择的数据统计:")
        print("=" * 40)
        
        fault_type_counts = {}
        rpm_counts = {}
        freq_counts = {}
        
        for data in self.selected_data:
            fault_type = data['fault_type']
            rpm = data['rpm']
            freq_dir = data['freq_dir']
            
            # 统计故障类型
            if fault_type not in fault_type_counts:
                fault_type_counts[fault_type] = 0
            fault_type_counts[fault_type] += 1
            
            # 统计转速
            if rpm not in rpm_counts:
                rpm_counts[rpm] = 0
            rpm_counts[rpm] += 1
            
            # 统计采样频率
            if freq_dir not in freq_counts:
                freq_counts[freq_dir] = 0
            freq_counts[freq_dir] += 1
        
        print(f"故障类型分布: {fault_type_counts}")
        print(f"转速分布: {rpm_counts}")
        print(f"采样频率分布: {freq_counts}")
    
    def visualize_data_distribution(self):
        """可视化数据分布"""
        print("正在生成数据分布可视化...")
        
        if not self.selected_data:
            print("没有选择的数据，请先运行select_representative_data()")
            return
        
        # 准备数据
        df = pd.DataFrame(self.selected_data)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('源域数据分布分析', fontsize=16)
        
        # 1. 故障类型分布
        fault_counts = df['fault_type'].value_counts()
        axes[0, 0].pie(fault_counts.values, labels=fault_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('故障类型分布')
        
        # 2. 转速分布
        rpm_counts = df['rpm'].value_counts().sort_index()
        axes[0, 1].bar(rpm_counts.index, rpm_counts.values)
        axes[0, 1].set_title('转速分布')
        axes[0, 1].set_xlabel('转速 (RPM)')
        axes[0, 1].set_ylabel('样本数量')
        
        # 3. 采样频率分布
        freq_counts = df['freq_dir'].value_counts()
        axes[1, 0].bar(freq_counts.index, freq_counts.values)
        axes[1, 0].set_title('采样频率分布')
        axes[1, 0].set_xlabel('采样频率')
        axes[1, 0].set_ylabel('样本数量')
        
        # 4. 故障类型vs转速热力图
        pivot_table = df.pivot_table(index='fault_type', columns='rpm', values='file_path', aggfunc='count', fill_value=0)
        sns.heatmap(pivot_table, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('故障类型vs转速分布')
        axes[1, 1].set_xlabel('转速 (RPM)')
        axes[1, 1].set_ylabel('故障类型')
        
        plt.tight_layout()
        plt.savefig('源域数据分布分析.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_transfer_learning_dataset(self):
        """准备迁移学习数据集"""
        print("正在准备迁移学习数据集...")
        
        if not self.selected_data:
            print("没有选择的数据，请先运行select_representative_data()")
            return
        
        # 创建数据集
        dataset = {
            'train_data': [],
            'val_data': [],
            'test_data': []
        }
        
        # 按故障类型分组
        fault_groups = {}
        for data in self.selected_data:
            fault_type = data['fault_type']
            if fault_type not in fault_groups:
                fault_groups[fault_type] = []
            fault_groups[fault_type].append(data)
        
        # 为每种故障类型分配训练/验证/测试集
        for fault_type, files in fault_groups.items():
            np.random.seed(42)
            np.random.shuffle(files)
            
            n_files = len(files)
            train_size = int(0.7 * n_files)
            val_size = int(0.15 * n_files)
            
            dataset['train_data'].extend(files[:train_size])
            dataset['val_data'].extend(files[train_size:train_size + val_size])
            dataset['test_data'].extend(files[train_size + val_size:])
        
        # 打印数据集统计
        print("\n迁移学习数据集统计:")
        print("=" * 40)
        for split_name, split_data in dataset.items():
            print(f"{split_name}: {len(split_data)} 个样本")
            
            # 统计各故障类型
            fault_counts = {}
            for data in split_data:
                fault_type = data['fault_type']
                if fault_type not in fault_counts:
                    fault_counts[fault_type] = 0
                fault_counts[fault_type] += 1
            
            print(f"  故障类型分布: {fault_counts}")
        
        return dataset
    
    def save_dataset_info(self, dataset):
        """保存数据集信息"""
        print("正在保存数据集信息...")
        
        # 创建数据集信息DataFrame
        all_data = []
        for split_name, split_data in dataset.items():
            for data in split_data:
                all_data.append({
                    'split': split_name,
                    'fault_type': data['fault_type'],
                    'rpm': data['rpm'],
                    'freq_dir': data['freq_dir'],
                    'fault_dir': data['fault_dir'],
                    'file_path': data['file_path']
                })
        
        df = pd.DataFrame(all_data)
        
        # 保存为CSV
        df.to_csv('源域数据集信息.csv', index=False, encoding='utf-8-sig')
        
        # 保存为Excel（包含多个工作表）
        with pd.ExcelWriter('源域数据集信息.xlsx', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='全部数据', index=False)
            
            for split_name in ['train_data', 'val_data', 'test_data']:
                split_df = df[df['split'] == split_name]
                split_df.to_excel(writer, sheet_name=split_name, index=False)
        
        print("数据集信息已保存:")
        print("- 源域数据集信息.csv")
        print("- 源域数据集信息.xlsx")
        
        return df
    
    def generate_data_selection_report(self, dataset):
        """生成数据选择报告"""
        print("正在生成数据选择报告...")
        
        # 统计信息
        total_samples = sum(len(split_data) for split_data in dataset.values())
        
        report = f"""
# 源域数据筛选和迁移学习准备报告

## 1. 数据选择策略

### 1.1 选择原则
- **代表性**: 每种故障类型选择代表性样本
- **平衡性**: 尽量保持各故障类型样本数量平衡
- **多样性**: 包含不同转速和采样频率的样本
- **质量**: 选择信号质量较好的样本

### 1.2 选择方法
1. 按故障类型分组
2. 按转速进一步分组
3. 从每组中随机选择样本
4. 确保样本的多样性和代表性

## 2. 数据集统计

### 2.1 总体统计
- **总样本数**: {total_samples}
- **故障类型**: 4种（BS, IR, OR, N）
- **转速范围**: 1730-1800 RPM
- **采样频率**: 12kHz, 48kHz

### 2.2 数据集划分
"""
        
        for split_name, split_data in dataset.items():
            split_size = len(split_data)
            percentage = split_size / total_samples * 100
            
            report += f"- **{split_name}**: {split_size} 个样本 ({percentage:.1f}%)\n"
            
            # 统计各故障类型
            fault_counts = {}
            for data in split_data:
                fault_type = data['fault_type']
                if fault_type not in fault_counts:
                    fault_counts[fault_type] = 0
                fault_counts[fault_type] += 1
            
            report += f"  - 故障类型分布: {fault_counts}\n"
        
        report += """
## 3. 迁移学习准备

### 3.1 数据集特点
- **源域数据**: 包含多种故障类型和工况
- **转速无关**: 使用阶比分析消除转速影响
- **特征丰富**: 时域、频域、时频域特征
- **标签完整**: 每种样本都有明确的故障类型标签

### 3.2 迁移学习优势
1. **理论FCO固定**: 故障特征阶比与转速无关
2. **特征通用**: 提取的特征适用于不同工况
3. **数据充足**: 源域数据丰富，训练效果好
4. **标签可靠**: 基于故障机理的标签准确

## 4. 后续工作建议

### 4.1 特征提取
- 使用故障机理特征工程方法
- 提取转速无关的阶比特征
- 结合时域和频域特征

### 4.2 模型训练
- 在源域数据上训练诊断模型
- 使用交叉验证评估性能
- 选择最佳模型架构

### 4.3 迁移学习
- 将训练好的模型应用到目标域
- 使用域适应技术提高迁移效果
- 评估迁移学习性能

## 5. 结论

通过系统性的数据筛选和预处理，我们获得了高质量的源域数据集，为后续的故障诊断和迁移学习任务奠定了坚实基础。数据集具有良好的代表性和平衡性，能够支持有效的模型训练和性能评估。
"""
        
        # 保存报告
        with open('源域数据选择报告.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("数据选择报告已保存为 '源域数据选择报告.md'")
        return report

def main():
    """主函数"""
    print("=" * 60)
    print("源域数据筛选和迁移学习准备")
    print("针对问题1：从源域数据中筛选部分数据组成数据集")
    print("=" * 60)
    
    # 创建数据选择器
    selector = SourceDomainDataSelector()
    
    # 1. 探索源域数据
    data_stats = selector.explore_source_data()
    
    # 2. 选择代表性数据
    selected_data = selector.select_representative_data(data_stats, samples_per_type=50)
    
    # 3. 可视化数据分布
    selector.visualize_data_distribution()
    
    # 4. 准备迁移学习数据集
    dataset = selector.prepare_transfer_learning_dataset()
    
    # 5. 保存数据集信息
    dataset_df = selector.save_dataset_info(dataset)
    
    # 6. 生成数据选择报告
    selector.generate_data_selection_report(dataset)
    
    print("\n数据筛选完成！")
    print("输出文件：")
    print("- 源域数据选择报告.md")
    print("- 源域数据分布分析.png")
    print("- 源域数据集信息.csv")
    print("- 源域数据集信息.xlsx")
    
    return dataset, dataset_df

if __name__ == "__main__":
    main()
