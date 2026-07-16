# 第四周学习周报

姓名： 黄子杰
日期： 7.10—7.16
Git/Gitee 仓库地址： https://github.com/OrbisLumen/ml-tenweeks

## 一、本周任务

1. 学习李宏毅机器学习课程 P29—P33，了解大模型、大资料与模型能力涌现之间的关系。
2. 学习图像生成模型相关课程 P35—P36、P39—P41，了解 VAE、Flow-based Model、Diffusion Model、GAN 和 Stable Diffusion 等生成模型的基本思想。
3. 继续学习 PyTorch 入门内容，并结合课程作业阅读 Self-attention、Transformer 和 Generative Model 相关代码。
4. 尝试进行 PaddlePaddle 包配置与初步实践。

## 二、完成情况

- 大模型与大资料
- 图像生成模型
- 课程作业材料整理
- PyTorch 与 PaddlePaddle

## 三、本周学习收获

1. 对“大模型 + 大资料”有了更具体的认识：模型能力不仅来自参数规模，也受到训练资料数量、质量和算力分配方式的影响。
2. 认识到模型规模扩大后可能出现非线性变化，能力提升并不一定平滑，因此评估大模型时不能只看小规模实验的趋势。
3. 初步建立了图像生成模型的比较框架：VAE 关注潜在表示与重建，Flow-based Model 强调可逆变换，Diffusion Model 依靠逐步去噪，GAN 依靠生成器和判别器的对抗学习。
4. 对 Stable Diffusion 等文本生成图像模型的整体流程有了初步理解，知道文本信息通常先经过 Text Encoder，再指导生成模型和 Decoder 生成图像。
5. 通过整理 HW4、HW5、HW6 的作业材料，进一步把 Self-attention、Transformer 与生成模型的理论学习和代码实践联系起来。
6. 也认识到周报需要和仓库中已有材料保持一致：PaddlePaddle 部分目前缺少可见实践记录，后续需要补充配置过程、示例代码和运行结果。
