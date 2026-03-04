Create a custom subagent with specific system prompt and name for reuse.

Usage:
- Define specialized agents with custom roles and boundaries
- Created agents can be referenced by name in the Task tool
- Use this when you need a specific agent type not covered by predefined agents
- The created agent configuration will be saved and can be used immediately

Example workflow:
1. Use CreateSubagent to define a specialized agent (e.g., 'code_reviewer')
2. Use the Task tool with agent='code_reviewer' to launch the created agent

---

创建一个具有特定系统提示词和名称的自定义子代理，以供复用。

用法：
- 定义具有自定义角色和边界的专门化代理
- 创建的代理可以在 Task 工具中通过名称引用
- 当你需要一种预定义代理未涵盖的特定代理类型时使用此工具
- 创建的代理配置会立即保存，并可以马上使用

示例工作流程：
1. 使用 CreateSubagent 定义一个专门化代理（例如 'code_reviewer'）
2. 使用 Task 工具并指定 agent='code_reviewer' 来启动已创建的代理
