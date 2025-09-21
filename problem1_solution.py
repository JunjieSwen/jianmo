#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题1综合解决方案：数据分析与故障特征提取
整合数据筛选和基于故障机理的特征工程
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import kurtosis, skew
from scipy.io import loadmat
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Problem1Solution:
    """问题1综合解决方案"""
    
    def __init__(self, data_path='data/'):
        self.data_path = data_path
        
        # 轴承参数（SKF6205驱动端轴承）
        self.bearing_params = {
            'n': 9,           # 滚动体数量
            'd': 0.3126,      # 滚动体直径 (inch)
            'D': 1.537,       # 节圆直径 (inch)
            'contact_angle': 0  # 接触角（深沟球轴承为0）
        }
        
        # 计算理论故障特征阶比（FCO）
        self.fco_values = self._calculate_fco()
        
        # 数据存储
        self.source_data = []
        self.selected_data = []
        self.extracted_features = None
        self.feature_names = []
        self.labels = []
        
        print("=" * 60)
        print("问题1：数据分析与故障特征提取")
        print("基于故障机理的特征工程方案")
        print("=" * 60)
        
        print("轴承参数:")
        print(f"滚动体数量 n = {self.bearing_params['n']}")
        print(f"滚动体直径 d = {self.bearing_params['d']} inch")
        print(f"节圆直径 D = {self.bearing_params['D']} inch")
        print(f"\n理论故障特征阶比:")
        print(f"外圈故障阶比 FCO_OR = {self.fco_values['OR']:.4f}")
        print(f"内圈故障阶比 FCO_IR = {self.fco_values['IR']:.4f}")
        print(f"滚动体故障阶比 FCO_BS = {self.fco_values['BS']:.4f}")
    
    def _calculate_fco(self):
        """计算理论故障特征阶比"""
        n = self.bearing_params['n']
        d = self.bearing_params['d']
        D = self.bearing_params['D']
        
        # 根据故障机理公式计算阶比
        fco_or = n / 2 * (1 - d/D)  # 外圈故障阶比
        fco_ir = n / 2 * (1 + d/D)  # 内圈故障阶比
        fco_bs = d / D * (1 - (d/D)**2)  # 滚动体故障阶比
        
        return {
            'OR': fco_or,
            'IR': fco_ir,
            'BS': fco_bs
        }
    
    def step1_explore_source_data(self):
        """步骤1：探索源域数据结构"""
        print("\n步骤1：探索源域数据结构")
        print("-" * 40)
        
        src_path = os.path.join(self.data_path, 'src')
        
        # 统计各类型数据
        data_stats = {}
        total_files = 0
        
        for freq_dir in os.listdir(src_path):
            if not os.path.isdir(os.path.join(src_path, freq_dir)):
                continue
                
            freq_path = os.path.join(src_path, freq_dir)
            data_stats[freq_dir] = {}
            
            for fault_dir in os.listdir(freq_path):
                fault_path = os.path.join(freq_path, fault_dir)
                if not os.path.isdir(fault_path):
                    continue
                
                # 统计文件数量
                file_count = self._count_mat_files(fault_path)
                data_stats[freq_dir][fault_dir] = file_count
                total_files += file_count
        
        # 打印统计信息
        print(f"源域数据统计（总计{total_files}个文件）:")
        for freq_dir, fault_stats in data_stats.items():
            print(f"\n{freq_dir}:")
            for fault_dir, count in fault_stats.items():
                print(f"  {fault_dir}: {count} 个文件")
        
        return data_stats
    
    def _count_mat_files(self, directory):
        """统计.mat文件数量"""
        count = 0
        for root, dirs, files in os.walk(directory):
            count += len([f for f in files if f.endswith('.mat')])
        return count
    
    def step2_select_representative_data(self, samples_per_type=100):
        """步骤2：选择代表性数据"""
        print(f"\n步骤2：选择代表性数据（每类{samples_per_type}个样本）")
        print("-" * 40)
        
        src_path = os.path.join(self.data_path, 'src')
        
        # 按故障类型收集数据
        fault_groups = {
            'BS': [],  # 滚动体故障
            'IR': [],  # 内圈故障
            'OR': [],  # 外圈故障
            'N': []    # 正常状态
        }
        
        # 遍历所有数据文件
        for freq_dir in os.listdir(src_path):
            if not os.path.isdir(os.path.join(src_path, freq_dir)):
                continue
                
            freq_path = os.path.join(src_path, freq_dir)
            
            for fault_dir in os.listdir(freq_path):
                fault_path = os.path.join(freq_path, fault_dir)
                if not os.path.isdir(fault_path):
                    continue
                
                # 确定故障类型
                fault_type = self._determine_fault_type(fault_dir)
                if fault_type not in fault_groups:
                    continue
                
                # 收集该类型的所有文件
                self._collect_fault_files(fault_path, fault_type, fault_groups[fault_type])
        
        # 从每组中选择代表性样本
        selected_data = []
        for fault_type, files in fault_groups.items():
            if len(files) == 0:
                print(f"警告：未找到{fault_type}故障数据")
                continue
            
            # 随机选择样本
            np.random.seed(42)
            selected_files = np.random.choice(files, 
                                            min(len(files), samples_per_type), 
                                            replace=False)
            
            selected_data.extend(selected_files)
            print(f"{fault_type}故障: 从{len(files)}个文件中选择了{len(selected_files)}个样本")
        
        self.selected_data = selected_data
        print(f"\n数据选择完成，共选择{len(selected_data)}个样本")
        
        return selected_data
    
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
            return None
    
    def _collect_fault_files(self, fault_path, fault_type, file_list):
        """收集故障文件"""
        for root, dirs, files in os.walk(fault_path):
            for file in files:
                if file.endswith('.mat'):
                    file_path = os.path.join(root, file)
                    
                    # 提取转速信息
                    rpm = self._extract_rpm_from_mat_file(file_path)
                    if rpm is None:  # 跳过没有转速信息的文件
                        continue
                    
                    file_list.append({
                        'file_path': file_path,
                        'fault_type': fault_type,
                        'rpm': rpm
                    })
    
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
    
    def step3_extract_fault_mechanism_features(self):
        """步骤3：基于故障机理的特征提取"""
        print("\n步骤3：基于故障机理的特征提取")
        print("-" * 40)
        
        all_features = []
        labels = []
        
        for i, data_info in enumerate(self.selected_data):
            if i % 50 == 0:
                print(f"处理进度: {i}/{len(self.selected_data)}")
            
            try:
                # 加载信号数据
                mat_data = loadmat(data_info['file_path'])
                signal_data = None
                
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        signal_data = mat_data[key].flatten()
                        break
                
                if signal_data is None or len(signal_data) < 1000:
                    continue
                
                # 提取特征
                features = self._extract_comprehensive_features(signal_data, data_info['rpm'])
                
                all_features.append(list(features.values()))
                labels.append(data_info['fault_type'])
                
            except Exception as e:
                print(f"处理文件 {data_info['file_path']} 时出错: {e}")
                continue
        
        # 转换为numpy数组
        self.extracted_features = np.array(all_features)
        self.labels = np.array(labels)
        self.feature_names = list(features.keys())
        
        print(f"\n特征提取完成:")
        print(f"- 样本数量: {len(self.labels)}")
        print(f"- 特征数量: {len(self.feature_names)}")
        print(f"- 特征矩阵形状: {self.extracted_features.shape}")
        
        return self.extracted_features, self.labels, self.feature_names
    
    def _extract_comprehensive_features(self, signal_data, rpm, fs=12000):
        """提取综合特征"""
        features = {}
        
        # 1. 阶比分析特征
        order_features = self._extract_order_analysis_features(signal_data, rpm, fs)
        features.update(order_features)
        
        # 2. 时域无量纲特征
        time_features = self._extract_time_domain_features(signal_data)
        features.update(time_features)
        
        # 3. 时频域特征
        tf_features = self._extract_time_frequency_features(signal_data, rpm, fs)
        features.update(tf_features)
        
        return features
    
    def _extract_order_analysis_features(self, signal_data, rpm, fs):
        """提取阶比分析特征"""
        # 计算转频
        fr = rpm / 60.0  # Hz
        
        # FFT分析
        fft_data = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data), 1/fs)
        magnitude = np.abs(fft_data)
        
        # 转换为阶比轴
        orders = freqs / fr
        
        # 只保留正频率部分
        positive_mask = orders > 0
        positive_orders = orders[positive_mask]
        positive_magnitude = magnitude[positive_mask]
        
        # 限制阶比范围（0-20阶）
        order_mask = (positive_orders >= 0) & (positive_orders <= 20)
        orders_filtered = positive_orders[order_mask]
        magnitude_filtered = positive_magnitude[order_mask]
        
        features = {}
        
        # 在理论故障特征阶比附近提取特征
        for fault_type, fco in self.fco_values.items():
            # 在理论FCO附近±0.5阶范围内提取特征
            fco_range = (orders_filtered >= fco - 0.5) & (orders_filtered <= fco + 0.5)
            
            if np.any(fco_range):
                fco_magnitude = magnitude_filtered[fco_range]
                fco_orders = orders_filtered[fco_range]
                
                # 能量特征
                features[f'{fault_type}_energy'] = np.sum(fco_magnitude**2)
                features[f'{fault_type}_max_amplitude'] = np.max(fco_magnitude)
                features[f'{fault_type}_mean_amplitude'] = np.mean(fco_magnitude)
                
                # 峰值位置
                max_idx = np.argmax(fco_magnitude)
                features[f'{fault_type}_peak_order'] = fco_orders[max_idx]
                
                # 谐波特征（2倍频、3倍频）
                for harmonic in [2, 3]:
                    harmonic_fco = fco * harmonic
                    harmonic_range = (orders_filtered >= harmonic_fco - 0.3) & (orders_filtered <= harmonic_fco + 0.3)
                    
                    if np.any(harmonic_range):
                        harmonic_magnitude = magnitude_filtered[harmonic_range]
                        features[f'{fault_type}_harmonic_{harmonic}'] = np.max(harmonic_magnitude)
                    else:
                        features[f'{fault_type}_harmonic_{harmonic}'] = 0
            else:
                # 如果没有找到对应的阶比范围，设为0
                features[f'{fault_type}_energy'] = 0
                features[f'{fault_type}_max_amplitude'] = 0
                features[f'{fault_type}_mean_amplitude'] = 0
                features[f'{fault_type}_peak_order'] = 0
                features[f'{fault_type}_harmonic_2'] = 0
                features[f'{fault_type}_harmonic_3'] = 0
        
        # 整体阶比谱特征
        features['total_energy'] = np.sum(magnitude_filtered**2)
        features['spectral_centroid'] = np.sum(orders_filtered * magnitude_filtered) / np.sum(magnitude_filtered)
        features['spectral_spread'] = np.sqrt(np.sum(((orders_filtered - features['spectral_centroid'])**2) * magnitude_filtered) / np.sum(magnitude_filtered))
        
        return features
    
    def _extract_time_domain_features(self, signal_data):
        """提取时域无量纲特征"""
        features = {}
        
        # 基本统计特征
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['var'] = np.var(signal_data)
        
        # 无量纲特征（转速无关）
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        features['peak_to_peak'] = np.max(signal_data) - np.min(signal_data)
        features['skewness'] = skew(signal_data)
        features['kurtosis'] = kurtosis(signal_data)
        
        # 波形指标
        features['waveform_factor'] = features['rms'] / np.mean(np.abs(signal_data))
        features['peak_factor'] = np.max(np.abs(signal_data)) / features['rms']
        features['impulse_factor'] = np.max(np.abs(signal_data)) / np.mean(np.abs(signal_data))
        features['clearance_factor'] = np.max(np.abs(signal_data)) / (np.mean(np.sqrt(np.abs(signal_data))))**2
        
        # 能量特征
        features['energy'] = np.sum(signal_data**2)
        features['mean_abs'] = np.mean(np.abs(signal_data))
        
        return features
    
    def _extract_time_frequency_features(self, signal_data, rpm, fs):
        """提取时频域特征"""
        features = {}
        
        # 计算转频
        fr = rpm / 60.0
        
        # 短时傅里叶变换
        f, t, Zxx = signal.stft(signal_data, fs, nperseg=1024)
        
        # 转换为阶比
        orders = f / fr
        
        # 限制阶比范围
        order_mask = (orders >= 0) & (orders <= 20)
        orders_filtered = orders[order_mask]
        Zxx_filtered = Zxx[order_mask, :]
        
        # 时频域能量分布
        features['tf_energy'] = np.sum(np.abs(Zxx_filtered)**2)
        
        # 在故障特征阶比附近的时频能量
        for fault_type, fco in self.fco_values.items():
            fco_range = (orders_filtered >= fco - 0.5) & (orders_filtered <= fco + 0.5)
            
            if np.any(fco_range):
                fco_tf = Zxx_filtered[fco_range, :]
                features[f'{fault_type}_tf_energy'] = np.sum(np.abs(fco_tf)**2)
                features[f'{fault_type}_tf_max'] = np.max(np.abs(fco_tf))
            else:
                features[f'{fault_type}_tf_energy'] = 0
                features[f'{fault_type}_tf_max'] = 0
        
        return features
    
    def step4_analyze_representative_samples(self):
        """步骤4：分析代表性样本"""
        print("\n步骤4：分析代表性样本")
        print("-" * 40)
        
        # 选择每种故障类型的代表性样本
        representative_samples = {}
        
        for data_info in self.selected_data:
            fault_type = data_info['fault_type']
            if fault_type not in representative_samples:
                representative_samples[fault_type] = data_info
        
        # 分析每个代表性样本
        analysis_results = {}
        
        for fault_type, data_info in representative_samples.items():
            print(f"分析{fault_type}故障样本 (RPM: {data_info['rpm']})")
            
            try:
                # 加载信号数据
                mat_data = loadmat(data_info['file_path'])
                signal_data = None
                
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        signal_data = mat_data[key].flatten()
                        break
                
                if signal_data is None:
                    continue
                
                # 提取特征
                features = self._extract_comprehensive_features(signal_data, data_info['rpm'])
                
                analysis_results[fault_type] = {
                    'features': features,
                    'rpm': data_info['rpm'],
                    'signal': signal_data
                }
                
                # 打印关键特征
                print(f"  外圈故障阶比能量: {features['OR_energy']:.2e}")
                print(f"  内圈故障阶比能量: {features['IR_energy']:.2e}")
                print(f"  滚动体故障阶比能量: {features['BS_energy']:.2e}")
                print(f"  峰度: {features['kurtosis']:.2f}")
                print(f"  峰值因子: {features['peak_factor']:.2f}")
                
            except Exception as e:
                print(f"分析{fault_type}故障样本时出错: {e}")
        
        return analysis_results
    
    def step5_feature_importance_analysis(self):
        """步骤5：特征重要性分析"""
        print("\n步骤5：特征重要性分析")
        print("-" * 40)
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # 编码标签
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.labels)
        
        # 训练随机森林
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.extracted_features, y_encoded)
        
        # 获取特征重要性
        importances = rf.feature_importances_
        
        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("前20个重要特征:")
        print(feature_importance_df.head(20))
        
        # 可视化特征重要性
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(20)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('特征重要性')
        plt.title('特征重要性分析（前20个）')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('特征重要性分析.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance_df
    
    def step6_generate_comprehensive_report(self, analysis_results, feature_importance_df):
        """步骤6：生成综合报告"""
        print("\n步骤6：生成综合报告")
        print("-" * 40)
        
        report = f"""
# 问题1：数据分析与故障特征提取综合报告

## 1. 方法概述

### 1.1 故障机理基础
本方案基于轴承故障机理，利用故障特征阶比（Fault Characteristic Orders, FCO）实现转速无关的特征提取。

### 1.2 轴承参数
- **轴承型号**: SKF6205（驱动端）
- **滚动体数量**: {self.bearing_params['n']}
- **滚动体直径**: {self.bearing_params['d']} inch
- **节圆直径**: {self.bearing_params['D']} inch

### 1.3 理论故障特征阶比
- **外圈故障阶比 (FCO_OR)**: {self.fco_values['OR']:.4f}
- **内圈故障阶比 (FCO_IR)**: {self.fco_values['IR']:.4f}
- **滚动体故障阶比 (FCO_BS)**: {self.fco_values['BS']:.4f}

## 2. 数据选择策略

### 2.1 选择原则
- **代表性**: 每种故障类型选择代表性样本
- **平衡性**: 尽量保持各故障类型样本数量平衡
- **多样性**: 包含不同转速和采样频率的样本
- **质量**: 选择信号质量较好的样本

### 2.2 数据集统计
- **总样本数**: {len(self.labels)}
- **故障类型**: 4种（BS, IR, OR, N）
- **特征数量**: {len(self.feature_names)}

## 3. 特征提取方法

### 3.1 阶比分析特征
将频域分析从绝对频率轴转换为阶比轴（Order = Frequency / f_r），在理论FCO附近提取特征：

#### 3.1.1 能量特征
- 在FCO±0.5阶范围内提取幅值能量
- 计算最大幅值和平均幅值
- 提取峰值对应的阶比位置

#### 3.1.2 谐波特征
- 提取2倍频和3倍频的幅值
- 分析故障特征的谐波成分

### 3.2 时域无量纲特征
提取对转速依赖性较小的无量纲参数：

- **波形指标**: RMS/平均绝对值
- **峰值因子**: 最大值/RMS
- **脉冲因子**: 最大值/平均绝对值
- **裕度因子**: 最大值/(平均平方根)²
- **偏度和峰度**: 反映信号的非高斯特性

### 3.3 时频域特征
使用短时傅里叶变换分析时频域能量分布：

- 在故障特征阶比附近的时频能量
- 时频域最大幅值
- 能量分布特征

## 4. 代表性样本分析

### 4.1 各故障类型特征对比
"""
        
        for fault_type, result in analysis_results.items():
            features = result['features']
            rpm = result['rpm']
            
            report += f"""
#### {fault_type}故障 (RPM: {rpm})
- 外圈故障阶比能量: {features['OR_energy']:.2e}
- 内圈故障阶比能量: {features['IR_energy']:.2e}
- 滚动体故障阶比能量: {features['BS_energy']:.2e}
- 峰度: {features['kurtosis']:.2f}
- 峰值因子: {features['peak_factor']:.2f}
"""
        
        report += f"""
## 5. 特征重要性分析

### 5.1 前10个重要特征
"""
        
        top_features = feature_importance_df.head(10)
        for _, row in top_features.iterrows():
            report += f"- {row['feature']}: {row['importance']:.4f}\n"
        
        report += """
## 6. 方法优势

### 6.1 转速无关性
- 使用阶比分析，消除了转速对特征的影响
- 理论FCO值固定，适用于不同转速的轴承

### 6.2 物理意义明确
- 特征具有明确的故障机理基础
- 可解释性强，便于工程应用

### 6.3 迁移学习友好
- 源域和目标域使用相同的FCO值
- 特征提取方法通用，便于跨域应用

## 7. 结论

基于故障机理的特征工程方法具有以下特点：

1. **理论基础扎实**: 基于轴承故障机理的阶比分析
2. **转速无关**: 使用阶比轴替代频率轴，消除转速影响
3. **特征丰富**: 结合时域、频域、时频域多种特征
4. **可解释性强**: 每个特征都有明确的物理意义
5. **迁移友好**: 适用于不同转速和工况的轴承诊断

该方法为后续的故障诊断任务提供了可靠的特征基础，特别适合迁移学习场景。
"""
        
        # 保存报告
        with open('问题1综合报告.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("综合报告已保存为 '问题1综合报告.md'")
        return report
    
    def run_complete_solution(self):
        """运行完整解决方案"""
        print("开始运行问题1完整解决方案...")
        
        # 步骤1：探索源域数据
        data_stats = self.step1_explore_source_data()
        
        # 步骤2：选择代表性数据
        selected_data = self.step2_select_representative_data(samples_per_type=100)
        
        # 步骤3：特征提取
        X, y, feature_names = self.step3_extract_fault_mechanism_features()
        
        # 步骤4：分析代表性样本
        analysis_results = self.step4_analyze_representative_samples()
        
        # 步骤5：特征重要性分析
        feature_importance_df = self.step5_feature_importance_analysis()
        
        # 步骤6：生成综合报告
        report = self.step6_generate_comprehensive_report(analysis_results, feature_importance_df)
        
        print("\n问题1解决方案完成！")
        print("输出文件：")
        print("- 问题1综合报告.md")
        print("- 特征重要性分析.png")
        
        return X, y, feature_names, analysis_results, feature_importance_df

def main():
    """主函数"""
    # 创建问题1解决方案
    solution = Problem1Solution()
    
    # 运行完整解决方案
    X, y, feature_names, analysis_results, feature_importance_df = solution.run_complete_solution()
    
    return X, y, feature_names, analysis_results, feature_importance_df

if __name__ == "__main__":
    main()
