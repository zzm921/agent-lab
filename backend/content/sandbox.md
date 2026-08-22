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
## 为什么需要它

沙箱执行让 Agent 可以在隔离环境中运行代码、执行命令、操作文件，是 Harness"把执行交给环境"的核心组件。每次执行在独立容器中进行，资源受限，确保主系统安全；配合检查点可支持暂停 / 回滚 / 恢复。

## 怎么解决

安全是最大难点——如何防止逃逸、限制资源、隔离文件系统。我用 Docker 容器作为沙箱基础，每个任务启动临时容器，设置 CPU/内存限制，执行完即销毁；同时通过文件挂载实现工作区持久化，并强制 HITL 审批。

## 核心实现

```python
# 沙箱执行器
class SandboxExecutor:
    async def execute(self, command, timeout=30):
        container = await self.docker.containers.run(
            "sandbox:latest",
            command=["bash", "-c", command],
            detach=True,
            mem_limit="256m",
            cpu_quota=50000,
            network_disabled=True,
            volumes={workspace: {"bind": "/workspace"}},
        )

        try:
            result = await container.wait(
                timeout=timeout
            )
            logs = await container.logs()
            return {"exit_code": result["StatusCode"],
                    "output": logs.decode()}
        finally:
            await container.remove(force=True)
```

## 收益与边界

- Docker 容器隔离，进程 / 文件 / 网络三重隔离
- 资源限制：CPU、内存、执行时间，防止滥用
- 工作区挂载 + 检查点，支持多步任务与回滚
