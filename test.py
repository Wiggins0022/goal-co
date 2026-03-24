"""
GOAL
Copyright (c) 2024-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""
import argparse
import time
from types import SimpleNamespace
from learning.data_iterators import DataIterator
from learning.tester import Tester
from utils.exp import setup_experiment, setup_test_environment

if __name__ == "__main__":
    # =========================================================================
    # 参数配置区
    # =========================================================================

    # 模型路径配置
    PRETRAINED_MODEL_PATH = "./pretrained/single_task/tsp.best"
    SMALL_MODEL_PATH = "./pretrained/checkpoint-partition-300.pt"

    # 问题类型配置
    PROBLEM_TYPE = ["tsp"]  # 解决的问题类型

    # 数据集路径配置
    TEST_DATASET_PATH = ["data/atsp100_test.npz"]  # 测试数据路径

    # 评估参数配置
    BEAM_SIZE = 1  # 束搜索大小，1表示贪心解码
    KNN_NUM = -1  # 解码时使用的KNN数量，-1表示不使用

    # 设备与批次配置
    TEST_BATCH_SIZE = 128  # 测试批次大小
    SEED = 42  # 随机种子

    # 输出配置
    OUTPUT_DIR = "output/"  # 输出目录

    # =========================================================================
    # 创建参数命名空间（模拟命令行参数）
    # =========================================================================
    args = argparse.Namespace(
        # --- 网络参数 ---
        dim_node_idx=1,  # 节点随机索引维度
        dim_emb=128,  # 嵌入维度
        dim_ff=512,  # 前馈网络维度
        num_layers=9,  # 编码器层数
        num_heads=8,  # 注意力头数
        activation_ff="relu",  # 前馈网络激活函数
        node_feature_low_dim=8,  # 节点原型特征维度
        edge_feature_low_dim=4,  # 边原型特征维度
        activation_edge_adapter="relu",  # 边适配器激活函数

        # --- 小模型参数 ---
        embedding_dim=64,
        depth=12,
        feats=2,
        k_sparse=50,
        edge_feats=2,

        # --- 数据参数 ---
        problems=PROBLEM_TYPE,
        test_datasets=TEST_DATASET_PATH,
        output_dir=OUTPUT_DIR,

        # --- 任务与日志 ---
        job_id=0,

        # --- 模型加载 ---
        pretrained_model=PRETRAINED_MODEL_PATH,
        small_model=SMALL_MODEL_PATH,

        # --- 评估参数 ---
        beam_size=BEAM_SIZE,
        knns=KNN_NUM,

        # --- 通用参数 ---
        seed=SEED,
        test_batch_size=TEST_BATCH_SIZE,
        test_datasets_size=None,  # 测试数据集大小限制（None表示全部）
        debug=False,
    )

    # =========================================================================
    # 主程序逻辑（保持不变）
    # =========================================================================

    # 设置实验环境
    setup_experiment(args)

    # 设置测试环境并加载网络
    net,small_model = setup_test_environment(args)

    # 创建数据迭代器
    data_iterator = DataIterator(args, ddp=False)

    # 创建测试器
    tester = Tester(args, net, small_model, data_iterator.test_datasets)

    # 加载预训练模型
    tester.load_model(args.pretrained_model,args.small_model)

    # 执行测试并计时
    start_time = time.time()
    tester.test()
    print("inference time", time.time() - start_time)