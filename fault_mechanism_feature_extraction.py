#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于故障机理的特征工程方案
针对问题1：数据分析与故障特征提取
实现转速无关的故障特征阶比分析
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

class FaultMechanismFeatureExtractor:
    """基于故障机理的特征提取器"""
    
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
    
    def load_source_data(self):
        """加载源域数据"""
        print("正在加载源域数据...")
        
        source_data = []
        src_path = os.path.join(self.data_path, 'src')
        
        # 遍历所有源域数据
        for freq_dir in os.listdir(src_path):
            if not os.path.isdir(os.path.join(src_path, freq_dir)):
                continue
                
            freq_path = os.path.join(src_path, freq_dir)
            
            for fault_dir in os.listdir(freq_path):
                fault_path = os.path.join(freq_path, fault_dir)
                if not os.path.isdir(fault_path):
                    continue
                
                # 确定故障类型和转速
                fault_type, rpm = self._parse_fault_info(fault_dir, freq_dir)
                
                if fault_type is None:
                    continue
                
                # 加载该类型的所有文件
                self._load_fault_data(fault_path, fault_type, rpm, source_data)
        
        print(f"源域数据加载完成，共{len(source_data)}个样本")
        return source_data
    
    def _parse_fault_info(self, fault_dir, freq_dir):
        """解析故障类型和转速信息"""
        # 确定故障类型
        if fault_dir.startswith('B'):
            fault_type = 'BS'  # 滚动体故障
        elif fault_dir.startswith('IR'):
            fault_type = 'IR'  # 内圈故障
        elif fault_dir.startswith('OR'):
            fault_type = 'OR'  # 外圈故障
        elif fault_dir == 'N':
            fault_type = 'N'   # 正常状态
        else:
            return None, None
        
        return fault_type, None  # 转速信息需要从文件中读取
    
    def _load_fault_data(self, fault_path, fault_type, rpm, source_data):
        """加载故障数据"""
        for root, dirs, files in os.walk(fault_path):
            for file in files:
                if file.endswith('.mat'):
                    try:
                        file_path = os.path.join(root, file)
                        mat_data = loadmat(file_path)
                        
                        # 从文件中提取RPM信息
                        file_rpm = None
                        if 'RPM' in mat_data:
                            rpm_value = mat_data['RPM']
                            if hasattr(rpm_value, 'item'):
                                file_rpm = rpm_value.item()
                            elif isinstance(rpm_value, (int, float)):
                                file_rpm = float(rpm_value)
                            else:
                                file_rpm = float(rpm_value.flatten()[0])
                        
                        # 如果没有RPM信息，跳过该文件
                        if file_rpm is None:
                            continue
                        
                        # 提取信号数据
                        for key in mat_data.keys():
                            if not key.startswith('__') and key != 'RPM':
                                signal_data = mat_data[key].flatten()
                                
                                if len(signal_data) > 1000:
                                    source_data.append({
                                        'signal': signal_data,
                                        'fault_type': fault_type,
                                        'rpm': file_rpm,
                                        'file_path': file_path
                                    })
                                    break  # 只取第一个信号数据
                    except Exception as e:
                        print(f"加载文件 {file} 时出错: {e}")
    
    def extract_order_analysis_features(self, signal_data, rpm, fs=12000):
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
        
        # 1. 理论故障特征阶比附近的能量
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
        
        # 2. 整体阶比谱特征
        features['total_energy'] = np.sum(magnitude_filtered**2)
        features['spectral_centroid'] = np.sum(orders_filtered * magnitude_filtered) / np.sum(magnitude_filtered)
        features['spectral_spread'] = np.sqrt(np.sum(((orders_filtered - features['spectral_centroid'])**2) * magnitude_filtered) / np.sum(magnitude_filtered))
        
        # 3. 阶比谱的峰值特征
        peaks, _ = signal.find_peaks(magnitude_filtered, height=np.max(magnitude_filtered)*0.1)
        if len(peaks) > 0:
            features['peak_count'] = len(peaks)
            features['dominant_order'] = orders_filtered[peaks[np.argmax(magnitude_filtered[peaks])]]
        else:
            features['peak_count'] = 0
            features['dominant_order'] = 0
        
        return features, orders_filtered, magnitude_filtered
    
    def extract_time_domain_features(self, signal_data):
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
    
    def extract_time_frequency_features(self, signal_data, rpm, fs=12000):
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
    
    def analyze_representative_samples(self, source_data):
        """分析代表性样本"""
        print("正在分析代表性样本...")
        
        # 选择每种故障类型的代表性样本
        representative_samples = {}
        
        for sample in source_data:
            fault_type = sample['fault_type']
            if fault_type not in representative_samples:
                representative_samples[fault_type] = sample
        
        # 分析每个代表性样本
        analysis_results = {}
        
        for fault_type, sample in representative_samples.items():
            print(f"\n分析{fault_type}故障样本 (RPM: {sample['rpm']})")
            
            signal_data = sample['signal']
            rpm = sample['rpm']
            
            # 提取各种特征
            order_features, orders, magnitude = self.extract_order_analysis_features(signal_data, rpm)
            time_features = self.extract_time_domain_features(signal_data)
            tf_features = self.extract_time_frequency_features(signal_data, rpm)
            
            # 合并所有特征
            all_features = {**order_features, **time_features, **tf_features}
            
            analysis_results[fault_type] = {
                'features': all_features,
                'orders': orders,
                'magnitude': magnitude,
                'rpm': rpm,
                'signal': signal_data
            }
            
            # 打印关键特征
            print(f"  外圈故障阶比能量: {order_features['OR_energy']:.2e}")
            print(f"  内圈故障阶比能量: {order_features['IR_energy']:.2e}")
            print(f"  滚动体故障阶比能量: {order_features['BS_energy']:.2e}")
            print(f"  峰度: {time_features['kurtosis']:.2f}")
            print(f"  峰值因子: {time_features['peak_factor']:.2f}")
        
        return analysis_results
    
    def visualize_order_analysis(self, analysis_results):
        """可视化阶比分析结果"""
        print("正在生成阶比分析可视化...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('基于故障机理的阶比分析', fontsize=16)
        
        # 绘制每种故障类型的阶比谱
        fault_types = ['OR', 'IR', 'BS', 'N']
        colors = ['red', 'blue', 'green', 'black']
        
        for i, (fault_type, color) in enumerate(zip(fault_types, colors)):
            ax = axes[i//2, i%2]
            
            if fault_type in analysis_results:
                orders = analysis_results[fault_type]['orders']
                magnitude = analysis_results[fault_type]['magnitude']
                rpm = analysis_results[fault_type]['rpm']
                
                ax.plot(orders, magnitude, color=color, linewidth=1)
                ax.set_title(f'{fault_type}故障 - {rpm}RPM')
                ax.set_xlabel('阶比')
                ax.set_ylabel('幅值')
                ax.grid(True, alpha=0.3)
                
                # 标记理论故障特征阶比
                for fco_type, fco_value in self.fco_values.items():
                    ax.axvline(x=fco_value, color='red', linestyle='--', alpha=0.7, label=f'FCO_{fco_type}={fco_value:.2f}')
                
                ax.legend()
                ax.set_xlim(0, 15)
            else:
                ax.text(0.5, 0.5, f'无{fault_type}故障数据', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{fault_type}故障')
        
        plt.tight_layout()
        plt.savefig('故障机理阶比分析.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def extract_dataset_features(self, source_data):
        """对整个数据集进行特征提取"""
        print("正在对整个数据集进行特征提取...")
        
        all_features = []
        labels = []
        metadata = []
        
        for i, sample in enumerate(source_data):
            if i % 100 == 0:
                print(f"处理进度: {i}/{len(source_data)}")
            
            signal_data = sample['signal']
            fault_type = sample['fault_type']
            rpm = sample['rpm']
            
            # 提取所有特征
            order_features, _, _ = self.extract_order_analysis_features(signal_data, rpm)
            time_features = self.extract_time_domain_features(signal_data)
            tf_features = self.extract_time_frequency_features(signal_data, rpm)
            
            # 合并特征
            combined_features = {**order_features, **time_features, **tf_features}
            
            all_features.append(list(combined_features.values()))
            labels.append(fault_type)
            metadata.append({
                'rpm': rpm,
                'file_path': sample['file_path']
            })
        
        # 转换为numpy数组
        X = np.array(all_features)
        y = np.array(labels)
        
        # 创建特征名称
        feature_names = list(combined_features.keys())
        
        print(f"特征提取完成，共{len(feature_names)}个特征")
        print(f"数据集大小: {X.shape}")
        
        return X, y, feature_names, metadata
    
    def analyze_feature_importance(self, X, y, feature_names):
        """分析特征重要性"""
        print("正在分析特征重要性...")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # 编码标签
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # 训练随机森林
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y_encoded)
        
        # 获取特征重要性
        importances = rf.feature_importances_
        
        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n前20个重要特征:")
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
    
    def generate_feature_engineering_report(self, analysis_results, feature_importance_df):
        """生成特征工程报告"""
        print("正在生成特征工程报告...")
        
        report = f"""
# 基于故障机理的特征工程报告

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

## 2. 特征提取方法

### 2.1 阶比分析特征
将频域分析从绝对频率轴转换为阶比轴（Order = Frequency / f_r），在理论FCO附近提取特征：

#### 2.1.1 能量特征
- 在FCO±0.5阶范围内提取幅值能量
- 计算最大幅值和平均幅值
- 提取峰值对应的阶比位置

#### 2.1.2 谐波特征
- 提取2倍频和3倍频的幅值
- 分析故障特征的谐波成分

#### 2.1.3 整体阶比谱特征
- 总能量
- 频谱重心
- 频谱扩散度
- 峰值数量和主导阶比

### 2.2 时域无量纲特征
提取对转速依赖性较小的无量纲参数：

- **波形指标**: RMS/平均绝对值
- **峰值因子**: 最大值/RMS
- **脉冲因子**: 最大值/平均绝对值
- **裕度因子**: 最大值/(平均平方根)²
- **偏度和峰度**: 反映信号的非高斯特性

### 2.3 时频域特征
使用短时傅里叶变换分析时频域能量分布：

- 在故障特征阶比附近的时频能量
- 时频域最大幅值
- 能量分布特征

## 3. 代表性样本分析

### 3.1 各故障类型特征对比
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
## 4. 特征重要性分析

### 4.1 前10个重要特征
"""
        
        top_features = feature_importance_df.head(10)
        for _, row in top_features.iterrows():
            report += f"- {row['feature']}: {row['importance']:.4f}\n"
        
        report += """
## 5. 方法优势

### 5.1 转速无关性
- 使用阶比分析，消除了转速对特征的影响
- 理论FCO值固定，适用于不同转速的轴承

### 5.2 物理意义明确
- 特征具有明确的故障机理基础
- 可解释性强，便于工程应用

### 5.3 迁移学习友好
- 源域和目标域使用相同的FCO值
- 特征提取方法通用，便于跨域应用

## 6. 结论

基于故障机理的特征工程方法具有以下特点：

1. **理论基础扎实**: 基于轴承故障机理的阶比分析
2. **转速无关**: 使用阶比轴替代频率轴，消除转速影响
3. **特征丰富**: 结合时域、频域、时频域多种特征
4. **可解释性强**: 每个特征都有明确的物理意义
5. **迁移友好**: 适用于不同转速和工况的轴承诊断

该方法为后续的故障诊断任务提供了可靠的特征基础。
"""
        
        # 保存报告
        with open('故障机理特征工程报告.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("特征工程报告已保存为 '故障机理特征工程报告.md'")
        return report

def main():
    """主函数"""
    print("=" * 60)
    print("基于故障机理的特征工程方案")
    print("针对问题1：数据分析与故障特征提取")
    print("=" * 60)
    
    # 创建特征提取器
    extractor = FaultMechanismFeatureExtractor()
    
    # 1. 加载源域数据
    source_data = extractor.load_source_data()
    
    if len(source_data) == 0:
        print("警告：未找到源域数据，请检查数据路径")
        return
    
    # 2. 分析代表性样本
    analysis_results = extractor.analyze_representative_samples(source_data)
    
    # 3. 可视化阶比分析结果
    extractor.visualize_order_analysis(analysis_results)
    
    # 4. 对整个数据集进行特征提取
    X, y, feature_names, metadata = extractor.extract_dataset_features(source_data)
    
    # 5. 分析特征重要性
    feature_importance_df = extractor.analyze_feature_importance(X, y, feature_names)
    
    # 6. 生成特征工程报告
    extractor.generate_feature_engineering_report(analysis_results, feature_importance_df)
    
    print("\n特征工程完成！")
    print("输出文件：")
    print("- 故障机理特征工程报告.md")
    print("- 故障机理阶比分析.png")
    print("- 特征重要性分析.png")
    
    # 返回提取的特征和标签，供后续使用
    return X, y, feature_names, metadata

if __name__ == "__main__":
    main()
