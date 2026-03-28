# edafinanceagent

一个面向零基础学习者的股票投资顾问 Agent 复现项目说明。

本仓库基于原始 `Finance` 项目整理，重点保留并讲清以下三部分能力：

- 多 Agent 股票分析系统：`LangGraph + ReAct + MCP`
- 金融数据工具化：基于 MCP 协议封装 A 股数据查询能力
- 金融新闻训练链路理解：数据清洗、SFT 样本构造、Tokenizer 与标签对齐验证

本项目特别适合以下环境：

- 使用 `WSL 2 Ubuntu`
- 没有 `NVIDIA GPU`
- 通过 `OpenRouter` 调用大模型 API

## 适合谁

- 想把 Agent 项目写进简历，但希望先真正跑通再写的人
- 想理解多 Agent、MCP、SFT/LoRA 训练链路的人
- 只有普通笔记本、没有独显，也想完成项目复现的人

## 当前已经验证可行的内容

- 在 `WSL 2 + OpenRouter` 环境下跑通主系统
- 成功生成股票分析 Markdown 报告
- 修复 `Baostock` 并发登录导致的“用户未登录”问题
- 用轻量模型验证训练链路中的：
  - 数据过滤
  - Prompt 构造
  - Tokenization
  - Label Masking
  - 长 Prompt 截断问题
  - `left padding` 下的真实 token 区间判断

## 仓库结构

```text
Finance/
├─ Financial-MCP-Agent/            # 多 Agent 主工程
├─ a-share-mcp-is-just-i-need/     # A 股 MCP 数据服务
├─ nasdaq_news_sentiment/          # 情感数据样本
├─ risk_nasdaq/                    # 风险数据样本
├─ data_process.py                 # 新闻清洗、去重脚本
├─ train_qwen_sentiment.py         # 情感训练脚本
├─ train_qwen_risk.py              # 风险训练脚本
├─ test_qwen_sentiment.py          # 情感测试脚本
├─ test_risk_model.py              # 风险测试脚本
└─ download.py                     # 下载原始大模型脚本
```

## 推荐阅读顺序

1. 先看零基础复现教程：
[`docs/零基础复现教程.md`](D:\stuffs\作业\run\居丽叶简历项目3：股票投资顾问Agent\Finance\docs\零基础复现教程.md)
2. 跑通多 Agent 主系统
3. 再看训练链路理解部分
4. 最后再考虑云端补 LoRA / GRPO / vLLM

## 当前最推荐的复现路径

### 路线 A：先跑主系统

- WSL 2 中创建 Python 虚拟环境
- 配置 OpenRouter
- 修正 MCP 路径
- 启动多 Agent 主系统
- 生成股票分析报告

### 路线 B：再做训练链路理解

- 检查本地 CSV 数据结构
- 跑数据清洗脚本
- 构造生成式 SFT 样本
- 用轻量模型验证 tokenizer 与 label mask 逻辑

## 重要说明

本仓库当前更适合做：

- 工程复现
- Agent 架构学习
- 数据链路理解
- SFT 训练思路验证

本仓库当前不建议在无 NVIDIA GPU 环境下直接做：

- `Qwen3-8B` 正式 LoRA 训练
- `vLLM` 本地部署
- `GRPO` 正式强化学习训练

这些部分更适合迁移到云端 GPU 环境继续做。

## 适合作为简历项目的表述

- 基于 `LangGraph + ReAct` 构建股票投资顾问多 Agent 系统，设计基本面、技术面、估值、新闻四类分析 Agent，并由总结 Agent 生成结构化 Markdown 报告
- 基于 `MCP` 协议封装 A 股金融数据工具，统一数据接口与 Schema，支撑多 Agent 自主调用行情、财报、分红、新闻等数据
- 在 `WSL 2 + OpenRouter` 环境下完成系统复现与调试，并完成金融新闻 SFT 训练链路理解验证，定位标签 token 对齐与长 Prompt 截断问题

