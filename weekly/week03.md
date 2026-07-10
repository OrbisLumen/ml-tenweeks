# 第三周学习周报

姓名： 黄子杰
日期： 7.03—7.09
Git/Gitee 仓库地址： https://github.com/OrbisLumen/ml-tenweeks

## 一、本周任务

1. 学习李宏毅机器学习课程 P23—P25，了解大型语言模型中 Finetuning 与 Prompting 两类使用方式。
2. 学习 Colab、Jupyter Notebook 和 PyTorch 的基础用法，阅读 P10—P12、P22、P28 对应课程作业的代码。
3. 尝试完成课程作业 HW1 Regression、HW2 Classification 和 HW3 CNN，并调整作业中的数据获取与文件路径。
4. 初步了解 PyTorch 的数据加载、模型训练、验证、测试和模型保存流程。

## 二、完成情况

- Finetuning 与 Prompting
  - 学习了通过增加 Head、Finetune 和 Adapter 改造预训练模型的基本方式；
  - 了解了 In-context Learning、Instruction Learning 和 Instruction-Tuning；
  - 整理了 Chain of Thought、Self-consistency、Least-to-Most 等 Prompting 方法；
  - 区分了使用自然语言的 Hard Prompt 和可训练的 Soft Prompt。
- PyTorch 基础学习
  - 了解了 PyTorch 中 Tensor、`Dataset`、`DataLoader`、`torch.nn` 和 `torch.optim` 的基本作用；
  - 梳理了训练、验证与测试的基本流程，以及 `zero_grad()`、`backward()`、`step()` 三个训练步骤；
  - 理解了 `model.eval()` 和 `torch.no_grad()` 在验证、测试阶段的作用；
  - 学习了使用 `state_dict` 保存和加载模型参数的方法。
- 课程作业代码学习与修改
  - 阅读并梳理了 HW1、HW2、HW3 的代码结构，了解回归、语音分类和 CNN 图像分类任务的基本实现流程；
  - 对 HW1 和 HW2 的代码进行了修改：HW1 调整了数据下载及读取路径；HW2 除调整数据路径外，还在模型中加入了 `BatchNorm1d` 和 `Dropout`；
  - 将 HW1、HW2、HW3 原本依赖当前运行目录或 Notebook 命令的数据获取方式，改为使用 `Path` 管理仓库内的数据目录，并在代码中完成下载和解压；
  - HW1 已成功运行，完成了数据读取、训练与模型保存，仓库中保留了模型文件和 TensorBoard 运行记录；
  - HW2 和 HW3 所需数据集的 Google Drive 资源失效，无法获取数据集，因此未继续进行数据预处理、模型训练和测试。其中 HW2 仅运行到数据获取阶段，HW3 未运行。

## 三、本周学习收获

1. 认识到 Finetuning 与 Prompting 代表了使用大型语言模型的两种不同思路：前者通过调整模型参数或增加模块适配任务，后者通过设计输入引导模型完成任务。
2. 对 PyTorch 训练流程有了初步的整体认识，能够将数据加载、模型定义、损失计算、反向传播、参数更新和模型保存联系起来。
3. 通过阅读三份课程作业，初步了解了不同任务虽然使用的数据和网络结构不同，但都遵循相似的训练与验证流程。
4. 在修改 Notebook 的过程中，进一步熟悉了使用 `pathlib.Path` 统一管理文件路径的方法，使代码不再强依赖特定的工作目录。
5. 认识到代码修改完成并不等于实验已经完成：HW2、HW3 因外部数据链接失效而无法运行，因此需要在周报中区分代码阅读、代码修改和实际运行结果。
