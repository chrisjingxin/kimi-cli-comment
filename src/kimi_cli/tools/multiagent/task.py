import asyncio
from pathlib import Path
from typing import override

# CallableTool2: 可调用工具的基类
# ToolError / ToolOk: 工具执行的失败/成功返回类型
# ToolReturnValue: 工具返回值的联合类型
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
# BaseModel, Field: Pydantic 模型基类和字段描述符，用于定义工具参数的 schema
from pydantic import BaseModel, Field

# MaxStepsReached: 当子代理执行步数超过上限时抛出的异常
# get_wire_or_none: 获取当前的 Wire（消息通信通道），如果不存在则返回 None
# run_soul: 核心执行函数，驱动一个 KimiSoul 实例完成一轮完整的交互循环
from kimi_cli.soul import MaxStepsReached, get_wire_or_none, run_soul
# Agent: 代理实体类  Runtime: 运行时上下文（包含会话、labor_market 等）
from kimi_cli.soul.agent import Agent, Runtime
# Context: 对话上下文类，管理会话历史并可持久化到文件
from kimi_cli.soul.context import Context
# KimiSoul: Kimi 的「灵魂」类，封装了代理的推理和工具调用循环
from kimi_cli.soul.kimisoul import KimiSoul
# get_current_tool_call_or_none: 获取当前正在执行的工具调用信息（用于追踪 tool_call_id）
from kimi_cli.soul.toolset import get_current_tool_call_or_none
# load_desc: 从 markdown 文件加载工具描述文本
from kimi_cli.tools.utils import load_desc
# next_available_rotation: 生成一个不与现有文件冲突的唯一文件路径（自动递增编号）
from kimi_cli.utils.path import next_available_rotation
# Wire: 消息通信通道，连接 soul（AI 侧）和 UI（用户侧）
from kimi_cli.wire import Wire
from kimi_cli.wire.types import (
    # ApprovalRequest: 需要用户审批的请求（如执行危险操作前的确认）
    ApprovalRequest,
    # ApprovalResponse: 用户对审批请求的响应
    ApprovalResponse,
    # SubagentEvent: 子代理事件的包装类型，用于将子代理的所有事件包装后转发到主 Wire
    SubagentEvent,
    # ToolCallRequest: 工具调用请求
    ToolCallRequest,
    # WireMessage: Wire 消息的通用基类型
    WireMessage,
)

# 子代理执行响应过短时，最多追加提问的次数上限
MAX_CONTINUE_ATTEMPTS = 1


# 当子代理的响应过于简短（< 200 字符）时，发送此追加提示词要求其提供更详细的总结
CONTINUE_PROMPT = """
Your previous response was too brief. Please provide a more comprehensive summary that includes:

1. Specific technical details and implementations
2. Complete code examples if relevant
3. Detailed findings and analysis
4. All important information that should be aware of by the caller
""".strip()


# ========== 工具参数定义 ==========
# Params 定义了 Task 工具接受的输入参数
class Params(BaseModel):
    # description: 任务的简短描述（3-5 个单词），用于日志记录和 UI 展示
    description: str = Field(description="A short (3-5 word) description of the task")
    # subagent_name: 要使用的子代理名称，必须是已注册在 labor_market 中的代理
    subagent_name: str = Field(
        description="The name of the specialized subagent to use for this task"
    )
    # prompt: 交给子代理执行的任务描述，必须包含所有必要的背景信息
    # 因为子代理无法看到主代理的上下文
    prompt: str = Field(
        description=(
            "The task for the subagent to perform. "
            "You must provide a detailed prompt with all necessary background information "
            "because the subagent cannot see anything in your context."
        )
    )


# ========== Task 工具实现 ==========
# Task 工具是多代理协作的核心，用于派生子代理执行具体任务
# 主代理通过调用此工具将任务委派给子代理，子代理在独立的上下文中工作
# 完成后将结果（最后一条消息）返回给主代理
class Task(CallableTool2[Params]):
    # 工具名称，LLM 在 function call 时使用此名称
    name: str = "Task"
    # 参数类型声明
    params: type[Params] = Params

    def __init__(self, runtime: Runtime):
        super().__init__(
            # 从 task.md 文件加载工具描述，并动态替换其中的 ${SUBAGENTS_MD} 占位符
            # 为可用子代理列表的 markdown 格式文本
            description=load_desc(
                Path(__file__).parent / "task.md",
                {
                    # 遍历 labor_market 中所有固定子代理（在 agent.yaml 中预定义的），
                    # 生成 "- `name`: description" 格式的列表
                    "SUBAGENTS_MD": "\n".join(
                        f"- `{name}`: {desc}"
                        for name, desc in runtime.labor_market.fixed_subagent_descs.items()
                    ),
                },
            ),
        )
        # 保存 labor_market 引用，用于查找子代理
        self._labor_market = runtime.labor_market
        # 保存会话引用，用于获取上下文文件路径
        self._session = runtime.session

    async def _get_subagent_context_file(self) -> Path:
        """Generate a unique context file path for subagent.
        为子代理生成一个唯一的上下文文件路径。
        子代理的上下文文件名基于主代理上下文文件名 + "_sub" 后缀，
        如果同名文件已存在则自动递增编号（如 _sub1, _sub2...）。
        """
        # 获取主代理的上下文文件路径
        main_context_file = self._session.context_file
        # 构造子代理上下文文件的基础名称（主文件名 + "_sub"）
        subagent_base_name = f"{main_context_file.stem}_sub"
        # 确保父目录存在
        main_context_file.parent.mkdir(parents=True, exist_ok=True)  # just in case
        # 使用 next_available_rotation 找到一个不冲突的唯一文件路径
        sub_context_file = await next_available_rotation(
            main_context_file.parent / f"{subagent_base_name}{main_context_file.suffix}"
        )
        assert sub_context_file is not None
        return sub_context_file

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        # 获取所有已注册的子代理（包括固定子代理和动态创建的子代理）
        subagents = self._labor_market.subagents

        # 检查指定名称的子代理是否存在
        if params.subagent_name not in subagents:
            return ToolError(
                message=f"Subagent not found: {params.subagent_name}",
                brief="Subagent not found",
            )
        # 获取目标子代理实例
        agent = subagents[params.subagent_name]
        try:
            # 运行子代理并获取结果
            result = await self._run_subagent(agent, params.prompt)
            return result
        except Exception as e:
            return ToolError(
                message=f"Failed to run subagent: {e}",
                brief="Failed to run subagent",
            )

    async def _run_subagent(self, agent: Agent, prompt: str) -> ToolReturnValue:
        """Run subagent with optional continuation for task summary.
        运行子代理，如果响应过短则自动追加提问以获取更详细的总结。

        核心流程：
        1. 获取当前的 Wire 通信通道和工具调用 ID
        2. 设置事件转发机制（将子代理事件包装为 SubagentEvent 转发给主 Wire）
        3. 为子代理创建独立的 Context（上下文）
        4. 创建 KimiSoul 实例并运行
        5. 提取子代理的最终响应作为返回值
        """
        # 获取当前（主代理）的 Wire 通信通道
        super_wire = get_wire_or_none()
        assert super_wire is not None
        # 获取当前正在执行的工具调用信息（即这个 Task 工具调用本身）
        current_tool_call = get_current_tool_call_or_none()
        assert current_tool_call is not None
        # 保存当前工具调用的 ID，用于标记子代理事件的来源
        current_tool_call_id = current_tool_call.id

        def _super_wire_send(msg: WireMessage) -> None:
            """将子代理的消息转发到主 Wire 通道。
            
            转发策略：
            - ApprovalRequest/ApprovalResponse/ToolCallRequest: 直接透传到主 Wire（根层级）
              因为这些是需要用户交互的请求，必须传递到最顶层
            - 其他消息: 包装为 SubagentEvent 后发送，标记来源 task_tool_call_id
              这样 UI 端可以将这些事件正确归类到对应的子代理任务下
            """
            if isinstance(msg, ApprovalRequest | ApprovalResponse | ToolCallRequest):
                # 审批请求和工具调用请求直接透传到根 Wire 层级
                # Requests should stay at the root wire level.
                super_wire.soul_side.send(msg)
                return

            # 其他所有消息用 SubagentEvent 包装后发送
            # SubagentEvent 携带 task_tool_call_id 以标识这个事件属于哪个 Task 调用
            event = SubagentEvent(
                task_tool_call_id=current_tool_call_id,
                event=msg,
            )
            super_wire.soul_side.send(event)

        async def _ui_loop_fn(wire: Wire) -> None:
            """子代理的 UI 事件循环函数。
            
            从子代理的 Wire UI 侧持续接收消息，并通过 _super_wire_send 转发到主 Wire。
            merge=True 表示合并来自多个来源的消息流。
            这个函数作为协程在后台持续运行，直到 run_soul 完成。
            """
            wire_ui = wire.ui_side(merge=True)
            while True:
                msg = await wire_ui.receive()
                _super_wire_send(msg)

        # 为子代理生成唯一的上下文文件路径（每个子代理任务都有独立的上下文文件）
        subagent_context_file = await self._get_subagent_context_file()
        # 创建全新的 Context 实例，子代理从零开始，没有任何历史对话
        context = Context(file_backend=subagent_context_file)
        # 创建 KimiSoul 实例，将 agent 配置和新 context 组合在一起
        soul = KimiSoul(agent, context=context)

        try:
            # 运行子代理的核心执行循环
            # 参数：soul（代理灵魂）, prompt（任务描述）, _ui_loop_fn（事件转发）, asyncio.Event()（取消信号，这里不取消）
            await run_soul(soul, prompt, _ui_loop_fn, asyncio.Event())
        except MaxStepsReached as e:
            # 子代理执行步数超过上限，返回错误提示
            return ToolError(
                message=(
                    f"Max steps {e.n_steps} reached when running subagent. "
                    "Please try splitting the task into smaller subtasks."
                ),
                brief="Max steps reached",
            )

        # 子代理运行异常时的通用错误信息
        _error_msg = (
            "The subagent seemed not to run properly. Maybe you have to do the task yourself."
        )

        # ===== 验证子代理上下文的有效性 =====
        # 检查子代理是否产生了有效的对话历史，且最后一条消息来自 assistant（而非 user）
        if len(context.history) == 0 or context.history[-1].role != "assistant":
            return ToolError(message=_error_msg, brief="Failed to run subagent")

        # 提取子代理最后一条消息的文本内容作为返回值
        final_response = context.history[-1].extract_text(sep="\n")

        # ===== 响应过短时的自动追加提问机制 =====
        # 如果子代理的响应不足 200 个字符，且还有剩余的追加提问次数，
        # 则发送 CONTINUE_PROMPT 要求子代理提供更详细的总结
        n_attempts_remaining = MAX_CONTINUE_ATTEMPTS
        if len(final_response) < 200 and n_attempts_remaining > 0:
            # 使用同一个 soul 实例继续对话（上下文中已有之前的交互历史）
            await run_soul(soul, CONTINUE_PROMPT, _ui_loop_fn, asyncio.Event())

            # 再次验证上下文有效性
            if len(context.history) == 0 or context.history[-1].role != "assistant":
                return ToolError(message=_error_msg, brief="Failed to run subagent")
            # 更新为追加提问后的新响应
            final_response = context.history[-1].extract_text(sep="\n")

        # 返回子代理的最终响应内容给主代理
        return ToolOk(output=final_response)
