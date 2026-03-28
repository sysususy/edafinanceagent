# GitHub 上传指南

这份文档专门解决一件事：把当前项目整理成一个适合上传到 GitHub 的公开仓库。

本指南默认你希望把仓库命名为：

```text
edafinanceagent
```

## 1. 先确认你要上传哪一份代码

你现在实际上有两份项目副本：

1. Windows 工作区副本  
   路径示例：
   `D:\stuffs\作业\run\居丽叶简历项目3：股票投资顾问Agent\Finance`

2. WSL 中实际跑通的副本  
   路径示例：
   `~/projects/Finance`

推荐做法：

- 如果你后续还要继续在 WSL 里运行和调试，就优先上传 `~/projects/Finance`
- 如果你希望直接从 Windows 图形界面管理文件，就先把 WSL 里确认有效的修改同步回 Windows，再上传 Windows 这份

本仓库当前新增的文档文件已经放在 Windows 工作区中，所以如果你最终决定上传 WSL 副本，请记得把这些文档同步过去。

## 2. 上传前建议保留什么

建议保留：

- `README.md`
- `docs/零基础复现教程.md`
- `docs/GitHub上传指南.md`
- 源代码目录
- 少量示例数据
- 必要的 `requirements.txt`

建议不要上传：

- 虚拟环境目录
- `.env`
- OpenRouter API Key
- 大量日志
- 批量生成的报告
- 本地下载的大模型权重

本仓库根目录下的 `.gitignore` 已经帮你屏蔽了这些常见内容。

## 3. 第一次初始化 Git 仓库

如果当前目录还不是 Git 仓库，在项目根目录执行：

```bash
git init
git branch -M main
```

然后查看状态：

```bash
git status
```

## 4. 配置 Git 用户信息

如果你这台机器还没有配置过 Git 用户名和邮箱，执行：

```bash
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub邮箱"
```

检查是否生效：

```bash
git config --global --list
```

## 5. 把文件加入暂存区

在项目根目录执行：

```bash
git add .
```

然后检查即将提交的内容：

```bash
git status
```

你要重点确认：

- `.env` 没有被加入
- `.venv-agent/` 和 `.venv-train/` 没有被加入
- `Financial-MCP-Agent/logs/` 没有被加入
- `Financial-MCP-Agent/reports/` 没有被加入

如果发现不该上传的文件被加入了，先不要提交，先修改 `.gitignore`。

## 6. 创建第一次提交

确认无误后执行：

```bash
git commit -m "init: add edafinanceagent reproduction docs and source code"
```

## 7. 在 GitHub 网站创建空仓库

打开 GitHub，新建一个仓库：

- Repository name: `edafinanceagent`
- Public 或 Private：按你的需要选择
- 不要勾选 `Add a README file`
- 不要勾选 `.gitignore`
- 不要勾选 `license`

创建完成后，GitHub 会给你一个远程仓库地址，格式通常像：

```bash
https://github.com/你的用户名/edafinanceagent.git
```

## 8. 绑定远程仓库

在本地项目根目录执行：

```bash
git remote add origin https://github.com/你的用户名/edafinanceagent.git
```

检查是否添加成功：

```bash
git remote -v
```

## 9. 第一次推送

执行：

```bash
git push -u origin main
```

如果 GitHub 让你登录：

- 使用浏览器登录即可
- 如果提示使用 Token，就用 GitHub Personal Access Token

## 10. 推送后检查什么

推送成功后，到 GitHub 页面检查：

1. `README.md` 是否正常显示
2. `docs/` 目录是否存在
3. `.env` 是否没有上传
4. 虚拟环境是否没有上传
5. 日志和报告是否没有上传

## 11. 以后更新仓库怎么做

以后每次修改后，只需要重复这三步：

```bash
git add .
git commit -m "docs: update tutorial"
git push
```

## 12. 最推荐的上传顺序

如果你是零基础，最推荐按这个顺序走：

1. 先在要上传的目录里执行 `git init`
2. 再执行 `git status`
3. 看清楚哪些文件会被上传
4. 再 `git add .`
5. 再次 `git status`
6. 没问题再 `git commit`
7. 最后创建 GitHub 仓库并 `git push`

## 13. 一个非常重要的提醒

你现在的 WSL 副本和 Windows 副本不一定完全一致。

所以在正式上传前，你最好先做一次最终确认：

- 你准备上传的那一份，是否包含最新文档
- 你准备上传的那一份，是否包含实际修复过的 `Baostock` 会话锁逻辑
- 你准备上传的那一份，是否真的能按教程运行

不要把“文档最新版”和“代码最新版”分散在两份不同目录里上传。

## 14. 最终建议

如果你想把这个仓库做成真正适合展示给老师、同学或面试官看的项目，建议最终做到这三点：

1. 首页 `README.md` 能说明项目目标和亮点
2. `docs/零基础复现教程.md` 能带别人跑通主流程
3. 上传到 GitHub 的代码目录，与实际验证成功的环境保持一致
