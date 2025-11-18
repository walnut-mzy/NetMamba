# SD-MKD 数据预处理模块

本模块实现了从原始加密流量（PCAP/CSV/二进制流）到 PyTorch 可直接使用的张量数据的端到端处理流水线。代码位于 `src/data_preprocessing/` 目录，可单独运行，也可与训练脚本联动。

## 目录与核心组件
- `flow_extractor.py`：按五元组聚合数据包、支持超时拆分流，并能解析 PCAP/CSV/二进制。
- `byte_aggregator.py`：Stride-based 字节聚合，将前 `N` 个数据包的载荷拼接/截断/填充为固定长度。
- `dataset.py`：`FlowDataset` 数据集与 `create_dataloaders` 便捷函数。
- `label_encoder.py`：标签编码、逆频率类别权重计算。
- `data_splitter.py`：训练/验证/测试划分、Stacking 专用划分、k 折交叉验证。
- `cache_manager.py`：基于配置的缓存读写。
- `statistics.py`：数据统计与可视化接口（类别分布、长度直方图等）。
- `validator.py`：基础质量检查（标签范围、空流）。
- `adapters.py`：数据集适配器模式示例（ISCXVPN2016/ISCXTor2016/USTCTFC2016）。
- `preprocessor.py`：整合所有模块，输出 `FlowDataset` 与 `LabelEncoder`。
- `preprocess_data.py`：命令行入口示例。

## 快速开始
1. 安装依赖（可选的 PCAP 解析需要 `scapy` 或 `dpkt`）：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行预处理示例：
   ```bash
   python -m data_preprocessing.preprocess_data --dataset ISCXVPN2016 --data_path ./raw --max_length 1500 --output ./cache
   ```

   命令会：
   - 选择对应适配器读取原始数据；
   - 进行 stride-based 字节聚合、归一化；
   - 自动缓存处理结果；
   - 打印基础统计信息。

3. 在训练脚本中直接导入：
   ```python
   from data_preprocessing.preprocessor import DataPreprocessor, PreprocessConfig

   config = PreprocessConfig(dataset="ISCXVPN2016", data_path="./raw")
   dataset, encoder = DataPreprocessor(config).process_dataset()
   ```

## 参数与可配置项
`PreprocessConfig` 支持的数据路径、最大长度、采样数据包数、划分比例、缓存目录等均可通过 CLI 或代码指定。

## 注意事项
- 本仓库默认提供适配器骨架，真实场景需在适配器中填充具体的文件扫描与标签逻辑。
- 对于超大数据集，建议结合 `CacheManager` 和 `numpy.memmap`/HDF5 进一步优化内存占用。
- 统计与可视化函数依赖 `matplotlib`，在无显示环境下可写入文件。

