from pathlib import Path

# CallableTool2: 可调用工具的基类，提供工具注册和调用的通用框架
# ToolError: 工具执行失败时的返回类型
# ToolOk: 工具执行成功时的返回类型
# ToolReturnValue: 工具返回值的联合类型（ToolOk | ToolError）
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
# BaseModel, Field: Pydantic 模型基类和字段描述符，用于定义工具参数的 schema
from pydantic import BaseModel, Field

# Agent: 代理实体类，封装了一个完整的 AI 代理（包含名称、系统提示词、工具集和运行时配置）
# Runtime: 运行时上下文，包含会话信息、labor_market（代理劳动力市场）等
from kimi_cli.soul.agent import Agent, Runtime
# KimiToolset: Kimi 的工具集合类，管理所有可用工具
from kimi_cli.soul.toolset import KimiToolset
# load_desc: 从 markdown 文件加载工具描述文本的辅助函数
from kimi_cli.tools.utils import load_desc


# ========== 工具参数定义 ==========
# Params 定义了 CreateSubagent 工具接受的输入参数
class Params(BaseModel):
    # name: 子代理的唯一名称标识符，后续通过 Task 工具调用时需要引用此名称
    name: str = Field(
        description=(
            "Unique name for this agent configuration (e.g., 'summarizer', 'code_reviewer'). "
            "This name will be used to reference the agent in the Task tool."
        )
    )
    # system_prompt: 子代理的系统提示词，定义其角色、能力和行为边界
    system_prompt: str = Field(
        description="System prompt defining the agent's role, capabilities, and boundaries."
    )


# ========== CreateSubagent 工具实现 ==========
# CreateSubagent 工具允许主代理在运行时动态创建新的子代理
# 创建的子代理会注册到 labor_market 中，后续可通过 Task 工具按名称调用
class CreateSubagent(CallableTool2[Params]):
    # 工具名称，LLM 在 function call 时使用此名称来调用该工具
    name: str = "CreateSubagent"
    # 工具描述，从 create.md 文件加载，用于告诉 LLM 何时/如何使用此工具
    description: str = load_desc(Path(__file__).parent / "create.md")
    # 参数类型声明，指向上面定义的 Params 类
    params: type[Params] = Params

    def __init__(self, toolset: KimiToolset, runtime: Runtime):
        super().__init__()
        # 保存工具集引用，创建的子代理会与父代理共享同一个工具集
        self._toolset = toolset
        # 保存运行时引用，用于访问 labor_market（代理注册中心）
        self._runtime = runtime

    async def __call__(self, params: Params) -> ToolReturnValue:
        # 检查同名子代理是否已存在，避免重复创建
        if params.name in self._runtime.labor_market.subagents:
            return ToolError(
                message=f"Subagent with name '{params.name}' already exists.",
                brief="Subagent already exists",
            )

        # 创建新的 Agent 实例
        subagent = Agent(
            name=params.name,                    # 使用用户指定的名称
            system_prompt=params.system_prompt,   # 使用用户指定的系统提示词
            toolset=self._toolset,                # 与父代理共享同一套工具（share the same toolset as the parent agent）
            runtime=self._runtime.copy_for_dynamic_subagent(),  # 复制一份运行时配置给子代理
        )
        # 将新创建的子代理注册到 labor_market 的动态子代理列表中
        self._runtime.labor_market.add_dynamic_subagent(params.name, subagent)
        # 返回成功信息，附带当前所有可用子代理的名称列表
        return ToolOk(
            output="Available subagents: " + ", ".join(self._runtime.labor_market.subagents.keys()),
            message=f"Subagent '{params.name}' created successfully.",
        )
