# 安全说明

[English](SECURITY.md)

PocketInfer 读取本地 JSON 并写入生成的 JSON。它不执行模型仓库代码、不下载
checkpoint，也不会求值配置字段。

核心编译器不应加入 `trust_remote_code` 或隐式网络请求。公开仓库确定安全
联系人后，请通过私密渠道报告安全问题。
