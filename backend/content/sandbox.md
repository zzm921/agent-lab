---
id: sandbox
name: 沙箱命令执行
shortDesc: 隔离沙箱中执行系统命令，支持代码运行和文件操作，安全可控。
icon: terminal
difficulty: adv
completeLevel: 75
tags: [Sandbox, Code-Exec, Security, Harness]
techFilters: [MCP, FastAPI]
accent: '#22d3a8'
enabledTools: [run_command]
---
## 概述

沙箱（Sandbox）是 Agent 的**隔离执行环境**：Agent 在沙箱内运行代码、执行命令、操作文件，但无法影响宿主机与主系统。它让 Agent 拥有真正的「动手能力」，是 Harness「把执行交给环境」的核心组件。

一句话：**沙箱内自由，沙箱外安全**——Agent 在里面随便折腾，宿主机器毫发无损。

## 为什么需要

Agent 的核心能力之一是执行代码和命令，但直接在本机执行存在致命风险：

- **失控成本**：一次 `rm -rf /`、一条恶意或错误的命令就能摧毁宿主环境；
- **Prompt 注入**：外部数据（网页、文件）可能携带恶意指令，诱导 Agent 执行危险操作；
- **数据泄露**：Agent 执行时可能触碰宿主密钥、凭据等敏感信息；
- **资源滥用**：无限循环、海量内存消耗会拖垮主系统。

沙箱把「不可信的执行」与「可信的宿主」物理隔离，让 Agent 可以放开手脚，也让人类敢于放手。

## 部署形态：本地 vs 云端

| 形态 | 隔离强度 | 成本 | 运维 | 适用 |
|------|---------|------|------|------|
| 本地进程级（本机子进程） | 弱（共享内核） | 最低 | 零 | 开发调试、临时兜底 |
| 本地容器（本机 Docker） | 中 | 低 | 中 | 单机验证 |
| **云端沙箱（单独服务器部署）** | 强（独立内核/虚拟化） | 按量计费 | 托管 | **生产首选** |

**建议：使用云端沙箱，单独服务器部署。**

- **安全彻底隔离**：云端沙箱运行在独立服务器 / 虚拟机 / 微虚拟机（microVM）上，与业务主系统完全物理隔离，即使沙箱被攻破也无法触达宿主业务数据；
- **生命周期解耦**：沙箱的创建 / 暂停 / 销毁由云端平台管理，本地服务无需为执行环境常驻资源；生命周期应绑定「任务」而非「连接」，任务结束即销毁；
- **弹性与成本**：按需创建、用完即毁、空闲自动回收，闲时不付费，突发可弹性扩容；
- **运维省心**：环境预置、镜像预热、补丁、扩容、可观测均由平台负责，避免「自己搭环境 + 调度逻辑」的高运维成本。

> 本地进程级执行只适合**无容器环境下的轻量兜底**（如本项目 `local` 后端），生产环境不应依赖——它共享宿主内核，隔离强度不足以对抗恶意代码。

## 隔离技术对比

| 技术 | 原理 | 隔离强度 | 启动速度 | 代表 |
|------|------|---------|---------|------|
| 容器（Docker） | 共享宿主内核，namespace + cgroup | 中 | 快（秒级） | E2B、Daytona |
| gVisor | 用户态内核拦截系统调用 | 较强 | 快 | Beam、Modal、Google |
| **微虚拟机 microVM** | 每个沙箱独立 guest 内核（硬件隔离） | **最强** | 中（亚秒级） | E2B、CodeSandbox、AWS Lambda |

选型判断：执行「自己的代码、自己的依赖」时 gVisor 已足够；执行「不可信的第三方代码、多租户对抗负载」时应选 microVM（每个沙箱独立内核，内核级漏洞也无法波及宿主）。

## 沙箱生命周期

沙箱是一个有状态的对象，完整生命周期如下：

```
创建 ──→ 就绪 ──→ 执行 ──→（闲置超时 / TTL 到期 / 任务结束）──→ 销毁
              │        │
              └── 暂停 ─┴──→ 恢复（状态保留，暂停不收费）
```

| 阶段 | 说明 |
|------|------|
| **创建** | 基于模板/镜像启动；用**资源池化 + 镜像预热 + 快照**把冷启动压到亚秒级~毫秒级 |
| **就绪** | 健康探测通过，可接收命令 |
| **执行** | 运行命令/代码，受超时与资源限制约束 |
| **闲置超时** | 一定时长无活动即自动释放（如 `sandboxIdleTimeoutInSeconds`），避免占着资源不干活 |
| **TTL / 生命周期上限** | 单实例最长存活时长（如 6 小时 / 24 小时 / 7 天），到期强制回收 |
| **暂停 / 恢复** | 保存状态（内存快照 / 文件系统快照），暂停期间不收费，恢复后继续；多步长任务的关键能力 |
| **销毁** | 任务结束、超时或进程退出时彻底销毁并回收资源，防止容器堆积遗留 |

**复用优化**：创建沙箱成本高，同一 Agent 循环内不宜逐条新建/销毁。可按「配置指纹」（镜像 / 服务端 / 工作目录 / 挂载点）缓存复用，仅当配置变化或空闲超时后重建。

伪代码：

```python
class SandboxLifecycle:
    def get_or_create(settings):
        # 按配置指纹复用：指纹一致且未空闲超时 → 复用；否则销毁旧沙箱重建
        fp = fingerprint(image, endpoint, work_dir, mount_target)
        return pool.get(fp) or self._create_and_cache(fp, settings)

    def execute(self, sandbox, command, timeout):
        return sandbox.commands.run(command, opts=RunCommandOpts(timeout=timeout))

    def pause(self, sandbox):   # 长任务暂停，状态保留、停止计费
        sandbox.pause()
    def resume(self, sandbox):
        sandbox.resume()

    def destroy(self, sandbox):  # 任务结束 / 空闲超时 / 进程退出时调用
        sandbox.destroy()

    # 进程退出兜底：atexit 遍历池中全部沙箱销毁，防止遗留
```

## 关键设计要点

1. **资源限制**：CPU / 内存 / 执行超时 / 输出上限，防止资源滥用与超长刷屏；
2. **网络策略**：默认禁网或白名单，断掉数据外泄通道；
3. **最小权限**：非 root、受限用户、只读系统分区，只暴露必要路径；
4. **内容防线**：危险命令黑名单（格式化 / 递归删除 / 关机重启）、敏感环境变量剔除、输出截断；
5. **强制人工审批（HITL）**：高危工具（如命令执行）调用前必须由人确认，无论审批策略如何；
6. **审计与可观测**：命令记录、操作日志、监控告警，出问题可追溯。

## 文件与数据策略

**建议：不保存任何文件到本地。**

- 命令产生的文件落在**沙箱 / 云端工作目录**，通过挂载或 API（如 `/api/sandbox/files`）列表与下载，随取随用；
- 需要持久化的产物显式导出到远端存储，宿主本地不落盘，避免敏感数据驻留宿主机；
- 沙箱会话结束销毁时文件一并清除，实现「数据不留痕」；
- 敏感环境变量（key / token / secret）在执行前剔除，命令接触不到宿主凭据。

## 本项目的做法

本项目把沙箱能力封装为一个 `run_command` 工具，接入各 Agent 模式（react / plan_execute / reflection），采用**双后端架构**：

- **默认：OpenSandbox 云端沙箱**（`sandbox_backend=opensandbox`）——连接用户独立 Docker 部署的 OpenSandbox 服务端，命令在远端容器沙箱内执行，提供容器隔离、网络策略与生命周期管理；本地业务服务仅通过官方 SDK 接入，不持有执行环境；
- **兜底：local 本机轻量沙箱**（`sandbox_backend=local`）——无 Docker 环境时在「持久化工作目录 + 最小环境 + 超时硬杀 + 输出上限」下于本机执行，共享宿主内核，仅作开发兜底，生产建议使用 opensandbox。

伪代码：

```python
def run_command(command, timeout=0):
    if reason := deny_reason(command):        # 危险命令黑名单，命中即拒绝
        return f"已拒绝执行：{reason}"
    timeout = timeout or settings.sandbox_timeout
    work_dir = resolve_work_dir(settings)     # 工作目录，沙箱内挂载到 /work
    if settings.sandbox_backend == "opensandbox":
        result = run_opensandbox(command, timeout, settings, work_dir)  # 云端容器沙箱
    else:
        result = run_sandboxed(command, timeout, work_dir)              # 本机轻量兜底
    return truncate(result, settings.sandbox_max_output)                # 输出上限
```

安全与生命周期落地点：

- **强制 HITL**：`run_command` 位于 `ALWAYS_APPROVE_TOOLS` 集合中，无论审批策略如何都必须人工确认命令本身；
- **危险命令黑名单**：`rm -rf /`、`format c:`、`mkfs`、`shutdown` 等特征命中即拒绝——与沙箱隔离构成两层防御；
- **敏感环境剔除**：含 key / token / secret / password / credential / auth 的环境变量不传入子进程；
- **超时硬杀 + 输出截断**：超时后清理整个进程树（Windows `taskkill /T /F`，POSIX 进程组 SIGKILL），输出超限截断；
- **沙箱复用池**：按配置指纹（镜像 / 服务端 / 工作目录 / 挂载点）缓存复用沙箱，空闲 > 300s 或指纹变化才销毁重建，进程退出时 `atexit` 全量清理；
- **工作目录持久化**：命令在沙箱挂载点内执行，相对路径文件落盘宿主机工作目录，`/api/sandbox/files` 提供列表与下载；
- **连接层异常兜底**：沙箱创建 / 执行连接层失败抛出可重试错误，由工具层透明重试（指数退避）处理。

与通用建议的对应关系：

| 通用建议 | 本项目做法 |
|---------|-----------|
| 云端沙箱、单独服务器部署 | OpenSandbox 后端（独立 Docker 部署的服务端） |
| 本地不落盘 | 文件写入挂载工作目录，经 `/api/sandbox/files` 导出，不散落业务系统 |
| 生命周期显式管理 | 复用池 + 空闲 TTL + 进程退出兜底清理 |
| 高危操作保留 HITL | `run_command` 强制人工审批 |

## 主流方案对比

| 平台 | 隔离 | 特点 | 部署 |
|------|------|------|------|
| E2B | Firecracker microVM | SDK 生态最大、上手最快 | 托管 |
| Daytona | 容器（可 Kata） | 冷启动最快（<100ms）、持久工作区、支持 Computer Use | 托管 / 自托管 |
| Modal | gVisor | 沙箱内支持 GPU，适合推理/训练 | 托管 |
| Beam | gVisor | 开源、BYOC（自带云/自有硬件） | 自托管/托管 |
| OpenSandbox | 可插拔（gVisor/Kata/Firecracker） | 开源、自托管、支持 Docker 部署（本项目默认后端） | 自托管 |
| 各大云厂商（阿里云 AIO、腾讯云 Agent Runtime 等） | microVM/容器 | 免运维、生命周期托管、毫秒级弹性 | 托管 |

## 最佳实践建议

1. **生产一律用云端沙箱、单独服务器部署**，与主业务系统物理隔离；
2. **本地不落盘**：文件进沙箱，产物显式导出；
3. **生命周期显式管理**：任务结束 / 超时 / 空闲即销毁，配 idle-timeout + TTL + 进程退出兜底清理；
4. **按任务而非连接管理生命周期**，支持暂停/恢复以承载长任务；
5. **隔离强度按威胁模型选**：不可信代码用 microVM，自有代码 gVisor 足够；
6. **高危操作保留 HITL**：即使沙箱隔离，命令执行前仍由人确认。

## 局限与权衡

- 隔离越强成本与启动时间越高：microVM 强于容器但更慢，需在安全与性能间取舍；
- 云端沙箱依赖外部服务，需考虑网络延迟与可用性（可做本地轻量兜底）；
- 状态型长任务需要快照/暂停恢复支持，纯短暂执行环境无法承接；
- 沙箱无法完全替代权限管理：HITL、白名单、审计仍需配套。
