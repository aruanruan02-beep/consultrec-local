# 本地咨询录音处理系统开发指南

## 核心命令

### 编译与构建

- **前端打包编译 (Node/pnpm)**：
  ```bash
  PATH="/Users/ruanruan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:$PATH" /Users/ruanruan/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm build
  ```

### 启动服务

- **启动本地 Python Web 服务**：
  ```bash
  "./ConsultRec Launcher.app/Contents/MacOS/consultrec-launcher"
  ```
- **查看进程状态**：
  ```bash
  ps ax | grep consultrec
  ```

---

## 核心设计规范与约束

### 1. 技术栈与架构
- **前端**：基于 Vite 构建的 React App (`web/src/` 编译输出到 `web/static/`)。UI 库使用 Arco Design，样式在 `web/src/styles.css`。
- **后端**：FastAPI 构建的 Python 本地 Web 服务，由 `consultrec/web.py` 提供服务接口，业务逻辑在 `consultrec/processor.py` 中编排。
- **数据结构**：数据保存在 `data_root`（可由用户自定义，并在设置页用原生 AppleScript 对话框选择），包括 Case metadata 和 Session 目录。

### 2. 核心逻辑约束
- **临床记录生成 (Summary / SOAP)**：
  - 后端解析大模型 JSON 数据时，不再强校验固定的 Key，实现全动态提取。
  - 前端渲染使用 `Object.keys()` 循环，大模型输出什么就展示什么，避免结构限死。
  - 支持“复制内容”按钮将 Summary/SOAP 一键复制到剪贴板。
- **说话人修改流程**：
  - 手动修改某一说话人时，弹出包含 **“批量修改所有 X 处原角色名”** 复选框的 Modal 对话框，由用户决定是单次修改还是全文批量替换。
  - 角色下拉框中的选项（如 `说话人 1`）基于全文当前存量数据动态计算。如果某角色在全文已被全数归并，其选项会自动从下拉列表中移除。
