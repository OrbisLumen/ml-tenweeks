# finetune

## 对预训练模型做改造
调整 BERT
- 加外挂（Head）
- 做微调（Finetune）
- 加插件（Adapter） 

# prompt

## In-context Learning
根据范例学习
- 有些论文显示范例标号的正误几乎不影响模型正确率
- 有些论文显示范例标号的正误在模型相当大时会影响模型

## Instruction Learning
根据题目叙述学习

Instruction-Tuning

## Chain of Thought (CoT)

Self-consistency
- 搭配 CoT 使答案稳定

Least-to-Most 
- 拆解难的数学问题

## 用机器来找 Prompt

Prompt
- Hard Prompt
  - 自然语言做prompt
- Soft Prompt
  - 类似于把 Adapter 放输入

How
- Using reinforcement learning
- Using an LM to find prompt
