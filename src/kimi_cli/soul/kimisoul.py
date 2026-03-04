from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import kosong
import tenacity
from kosong import StepResult
from kosong.chat_provider import (
    APIConnectionError,
    APIEmptyResponseError,
    APIStatusError,
    APITimeoutError,
)
from kosong.message import Message
from tenacity import RetryCallState, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from kimi_cli.llm import ModelCapability
from kimi_cli.skill import Skill, read_skill_text
from kimi_cli.skill.flow import Flow, FlowEdge, FlowNode, parse_choice
from kimi_cli.soul import (
    LLMNotSet,
    LLMNotSupported,
    MaxStepsReached,
    Soul,
    StatusSnapshot,
    wire_send,
)
from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.compaction import SimpleCompaction
from kimi_cli.soul.context import Context
from kimi_cli.soul.message import check_message, system, tool_result_to_message
from kimi_cli.soul.slash import registry as soul_slash_registry
from kimi_cli.soul.toolset import KimiToolset
from kimi_cli.tools.dmail import NAME as SendDMail_NAME
from kimi_cli.tools.utils import ToolRejectedError
from kimi_cli.utils.logging import logger
from kimi_cli.utils.slashcmd import SlashCommand, parse_slash_command_call
from kimi_cli.wire.file import WireFile
from kimi_cli.wire.types import (
    ApprovalRequest,
    ApprovalResponse,
    CompactionBegin,
    CompactionEnd,
    ContentPart,
    StatusUpdate,
    StepBegin,
    StepInterrupted,
    TextPart,
    ToolResult,
    TurnBegin,
    TurnEnd,
)

if TYPE_CHECKING:

    def type_check(soul: KimiSoul):
        _: Soul = soul


SKILL_COMMAND_PREFIX = "skill:"
FLOW_COMMAND_PREFIX = "flow:"
DEFAULT_MAX_FLOW_MOVES = 1000


type StepStopReason = Literal["no_tool_calls", "tool_rejected"]


### zjx： 步骤结果 - 单次 LLM 调用的结果
@dataclass(frozen=True, slots=True)
class StepOutcome:
    stop_reason: StepStopReason
    assistant_message: Message


type TurnStopReason = StepStopReason


### zjx：轮次结果 - 一次用户交互的完整结果
@dataclass(frozen=True, slots=True)
class TurnOutcome:
    stop_reason: TurnStopReason
    final_message: Message | None
    step_count: int


class KimiSoul:
    """The soul of Kimi Code CLI."""

    def __init__(
        self,
        agent: Agent,
        *,
        context: Context,
    ):
        """
        Initialize the soul.

        Args:
            agent (Agent): The agent to run.
            context (Context): The context of the agent.
        """
        self._agent = agent
        self._runtime = agent.runtime
        self._denwa_renji = agent.runtime.denwa_renji
        self._approval = agent.runtime.approval
        self._context = context
        self._loop_control = agent.runtime.config.loop_control
        self._compaction = SimpleCompaction()  # TODO: maybe configurable and composable

        for tool in agent.toolset.tools:
            if tool.name == SendDMail_NAME:
                self._checkpoint_with_user_message = True
                break
        else:
            self._checkpoint_with_user_message = False

        self._slash_commands = self._build_slash_commands()
        self._slash_command_map = self._index_slash_commands(self._slash_commands)

    @property
    def name(self) -> str:
        return self._agent.name

    @property
    def model_name(self) -> str:
        return self._runtime.llm.chat_provider.model_name if self._runtime.llm else ""

    @property
    def model_capabilities(self) -> set[ModelCapability] | None:
        if self._runtime.llm is None:
            return None
        return self._runtime.llm.capabilities

    @property
    def thinking(self) -> bool | None:
        """Whether thinking mode is enabled."""
        if self._runtime.llm is None:
            return None
        if thinking_effort := self._runtime.llm.chat_provider.thinking_effort:
            return thinking_effort != "off"
        return None

    @property
    def status(self) -> StatusSnapshot:
        return StatusSnapshot(
            context_usage=self._context_usage,
            yolo_enabled=self._approval.is_yolo(),
        )

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def runtime(self) -> Runtime:
        return self._runtime

    @property
    def context(self) -> Context:
        return self._context

    @property
    def _context_usage(self) -> float:
        if self._runtime.llm is not None:
            return self._context.token_count / self._runtime.llm.max_context_size
        return 0.0

    @property
    def wire_file(self) -> WireFile:
        return self._runtime.session.wire_file

    async def _checkpoint(self):
        await self._context.checkpoint(self._checkpoint_with_user_message)

    @property
    def available_slash_commands(self) -> list[SlashCommand[Any]]:
        return self._slash_commands

    async def run(self, user_input: str | list[ContentPart]):
        # Refresh OAuth tokens on each turn to avoid idle-time expirations.
        ### zjx： 1. 刷新 OAuth token
        ### zjx：在每一轮对话（Turn）开始前，静默检查并刷新 OAuth Token。这解决了长时间挂机后，用户突然提问却因 Token 过期而报错的糟糕体验。
        await self._runtime.oauth.ensure_fresh(self._runtime)

        ### zjx：2. 发送 TurnBegin 事件
        wire_send(TurnBegin(user_input=user_input))

        ### zjx：3. 解析用户输入
        ### zjx：将用户的原始输入封装成标准的 Message 对象
        user_message = Message(role="user", content=user_input)
        ### zjx：将用户输入的文本抽取出来
        text_input = user_message.extract_text(" ").strip()

        ### zjx：判断是否为斜杠指令的同时，直接将解析结果赋值给 command_call
        if command_call := parse_slash_command_call(text_input):
            ### zjx：根据指令名称（如 help, clear）去已注册的指令表里寻找对应的执行对象。
            command = self._find_slash_command(command_call.name)
            ### zjx：如果没有找到指令的话，提示错误
            if command is None:
                # this should not happen actually, the shell should have filtered it out
                wire_send(TextPart(text=f'Unknown slash command "/{command_call.name}".'))
            ### zjx：如果找到指令的话，就执行
            else:
                ret = command.func(self, command_call.args)
                ### zjx：Python 兼容性设计
                ### zjx：它允许指令回调函数既可以是普通同步函数（def），也可以是异步函数（async def）
                ### zjx：如果是异步函数，其返回值是一个 Awaitable（可等待对象），系统会自动对其进行 await 挂起执行。
                if isinstance(ret, Awaitable):
                    await ret

        ### zjx：如果不是斜杠指令，就是普通输入
        ### zjx：执行ralph loop
        elif self._loop_control.max_ralph_iterations != 0:
            ### zjx：以loop的方式循环执行
            runner = FlowRunner.ralph_loop(
                user_message,
                self._loop_control.max_ralph_iterations,
            )
            ### zjx：启动这个复杂的 Agent 工作流
            await runner.run(self, "")
        else:
            ### zjx：: 如果没有触发特殊指令，也没有开启高级的 Ralph 循环，就会走到这个兜底的默认分支：执行一个最基础的单轮对话逻辑 (_turn)。
            await self._turn(user_message)

        ### zjx：不管走了上述哪条路径，最后都向渲染端发送一个“回合结束”的事件
        wire_send(TurnEnd())

    async def _turn(self, user_message: Message) -> TurnOutcome:
        ### zjx：在执行任何逻辑前，先确认“大脑”（模型驱动）是否已挂载
        ### zjx：防御性编程
        if self._runtime.llm is None:
            raise LLMNotSet()

        ### zjx：会对比“用户输入的内容”与“当前模型的能力（capabilities）”
        ### zjx：例子：如果用户发了一张图片，但当前配置的模型不支持 Vision（视觉），这里会立即抛出 LLMNotSupported
        if missing_caps := check_message(user_message, self._runtime.llm.capabilities):
            raise LLMNotSupported(self._runtime.llm, list(missing_caps))

        ### zjx：它在处理新消息前先建立一个“检查点”
        await self._checkpoint()  # this creates the checkpoint 0 on first run

        ### zjx：将当前用户的消息压入会话历史
        await self._context.append_message(user_message)
        logger.debug("Appended user message to context")

        ### zjx：发起agent loop
        return await self._agent_loop()

    def _build_slash_commands(self) -> list[SlashCommand[Any]]:
        ### zjx：一个全局注册表，存放所有内置斜杠命令
        commands: list[SlashCommand[Any]] = list(soul_slash_registry.list_commands())
        seen_names = {cmd.name for cmd in commands}

        ### zjx：注册 Skill 斜杠命令
        for skill in self._runtime.skills.values():
            if skill.type not in ("standard", "flow"):
                continue
            name = f"{SKILL_COMMAND_PREFIX}{skill.name}"
            if name in seen_names:
                logger.warning(
                    "Skipping skill slash command /{name}: name already registered",
                    name=name,
                )
                continue
            commands.append(
                SlashCommand(
                    name=name,
                    func=self._make_skill_runner(skill),
                    description=skill.description or "",
                    aliases=[],
                )
            )
            seen_names.add(name)

        ### zjx：注册 Flow 专属命令
        for skill in self._runtime.skills.values():
            if skill.type != "flow":
                continue
            if skill.flow is None:
                logger.warning("Flow skill {name} has no flow; skipping", name=skill.name)
                continue
            command_name = f"{FLOW_COMMAND_PREFIX}{skill.name}"
            if command_name in seen_names:
                logger.warning(
                    "Skipping prompt flow slash command /{name}: name already registered",
                    name=command_name,
                )
                continue
            runner = FlowRunner(skill.flow, name=skill.name)
            commands.append(
                SlashCommand(
                    name=command_name,
                    func=runner.run,
                    description=skill.description or "",
                    aliases=[],
                )
            )
            seen_names.add(command_name)

        return commands

    @staticmethod
    def _index_slash_commands(
        commands: list[SlashCommand[Any]],
    ) -> dict[str, SlashCommand[Any]]:
        indexed: dict[str, SlashCommand[Any]] = {}
        for command in commands:
            indexed[command.name] = command
            for alias in command.aliases:
                indexed[alias] = command
        return indexed

    def _find_slash_command(self, name: str) -> SlashCommand[Any] | None:
        return self._slash_command_map.get(name)

    def _make_skill_runner(self, skill: Skill) -> Callable[[KimiSoul, str], None | Awaitable[None]]:
        async def _run_skill(soul: KimiSoul, args: str, *, _skill: Skill = skill) -> None:
            skill_text = await read_skill_text(_skill)
            if skill_text is None:
                wire_send(
                    TextPart(text=f'Failed to load skill "/{SKILL_COMMAND_PREFIX}{_skill.name}".')
                )
                return
            extra = args.strip()
            if extra:
                skill_text = f"{skill_text}\n\nUser request:\n{extra}"
            await soul._turn(Message(role="user", content=skill_text))

        _run_skill.__doc__ = skill.description
        return _run_skill

    async def _agent_loop(self) -> TurnOutcome:
        """The main agent loop for one run."""
        assert self._runtime.llm is not None
        if isinstance(self._agent.toolset, KimiToolset):
            ### zjx：加载MCP工具
            await self._agent.toolset.wait_for_mcp_tools()

        """
        ====================================================
        # Commentator: zjx
        # 在执行危险操作前，暂停下来问用户"你确定吗？"
        
        时间线 ──────────────────────────────────────────►

        Agent (_step)                    审批管道                      用户终端
            │                               │                            │
            │  调用工具 bash("rm ...")       │                            │
            │  ──► 需要审批！                │                            │
            │  放入 approval 队列 ─────────► │                            │
            │                               │                            │
            │  (Agent 在这里阻塞等待)        │  fetch_request() 取出请求  │
            │                               │                            │
            │                               │  wire_send(请求) ─────────►│
            │                               │                            │
            │                               │                            │ 用户看到:
            │                               │                            │ "执行 rm -rf?"
            │                               │                            │ 用户按下 [Y]
            │                               │                            │
            │                               │  ◄───── wire_request.wait()│
            │                               │         返回 "approved"    │
            │                               │                            │
            │                               │  resolve_request(approved) │
            │  ◄──── 审批通过，继续执行 ─────│                            │
            │                               │                            │
            │  bash 真正执行                 │  wire_send(审批响应) ─────►│
            │  返回结果给 LLM               │                            │ 显示 ✓ 已批准
            ▼                               ▼                            ▼
        ====================================================
        """
        async def _pipe_approval_to_wire():
            while True:
                request = await self._approval.fetch_request()
                # Here we decouple the wire approval request and the soul approval request.
                wire_request = ApprovalRequest(
                    id=request.id,
                    action=request.action,
                    description=request.description,
                    sender=request.sender,
                    tool_call_id=request.tool_call_id,
                    display=request.display,
                )
                wire_send(wire_request)
                # We wait for the request to be resolved over the wire, which means that,
                # for each soul, we will have only one approval request waiting on the wire
                # at a time. However, be aware that subagents (which have their own souls) may
                # also send approval requests to the root wire.
                resp = await wire_request.wait()
                self._approval.resolve_request(request.id, resp)
                wire_send(ApprovalResponse(request_id=request.id, response=resp))

        step_no = 0
        while True:
            step_no += 1
            if step_no > self._loop_control.max_steps_per_turn:
                raise MaxStepsReached(self._loop_control.max_steps_per_turn)

            ### zjx：发送当前步骤开始的事件
            wire_send(StepBegin(n=step_no))
            ### zjx：创建一个后台任务来运行前面定义的审批管道函数。这个任务会独立运行，持续处理审批请求，而主循环继续执行其他逻辑。
            approval_task = asyncio.create_task(_pipe_approval_to_wire())
            back_to_the_future: BackToTheFuture | None = None
            step_outcome: StepOutcome | None = None
            try:
                # compact the context if needed
                ### zjx：保留的上下文大小
                reserved = self._loop_control.reserved_context_size
                ### zjx：计算当前 token 计数加上保留大小是否超过 LLM 的最大上下文大小
                if self._context.token_count + reserved >= self._runtime.llm.max_context_size:
                    logger.info("Context too long, compacting...")
                    ### zjx：上下文压缩
                    await self.compact_context()

                logger.debug("Beginning step {step_no}", step_no=step_no)
                ### zjx：创建检查点
                await self._checkpoint()
                self._denwa_renji.set_n_checkpoints(self._context.n_checkpoints)

                """
                ====================================================
                # Commentator: zjx
                重点！！！ 调用核心的_step
                可能返回 None 或 StepOutcome
                在_step返回None且没有回溯的情况下，进入下一个循环
                ====================================================
                """
                step_outcome = await self._step()


            except BackToTheFuture as e:
                ### zjx：当 Agent 或工具检测到需要"撤销"当前操作并回到之前的状态时，抛出这个异常。
                ### zjx：例如：用户拒绝了某个操作的审批，Agent 需要回退。
                back_to_the_future = e
            except Exception:
                # any other exception should interrupt the step
                ### zjx：其他异常，通知 wire 层"步骤被中断"
                wire_send(StepInterrupted())
                # break the agent loop
                ### zjx：重新抛出异常，终止整个 Agent 循环
                raise
            finally:
                ### zjx：首先取消审批管道任务，停止向导线发送新的审批请求
                approval_task.cancel()  # stop piping approval requests to the wire
                ### zjx：忽略取消导致的 CancelledError（这是正常的）
                with suppress(asyncio.CancelledError):
                    try:
                        ### zjx：等待任务真正结束（优雅关闭）
                        await approval_task
                    except Exception:
                        ### zjx：如果审批任务本身出了其他错误，记录日志但不影响主流程。
                        logger.exception("Approval piping task failed")

            ### zjx：如果 _step() 返回了结果（没有抛异常）
            if step_outcome is not None:
                ### zjx：如果LLM 没有调用任何工具，说明它给出了最终回答 → 取出 assistant_message 作为最终消息
                final_message = (
                    step_outcome.assistant_message
                    if step_outcome.stop_reason == "no_tool_calls"
                    else None
                )
                return TurnOutcome(
                    stop_reason=step_outcome.stop_reason,
                    final_message=final_message,
                    step_count=step_no,
                )

            if back_to_the_future is not None:
                ### zjx：将上下文回退到指定的检查点状态（丢弃之后的所有消息）
                await self._context.revert_to(back_to_the_future.checkpoint_id)
                ### zjx：在回退后创建一个新的检查点。
                await self._checkpoint()
                ### zjx：追加回溯异常携带的新消息（通常是解释为什么要回溯的信息）
                await self._context.append_message(back_to_the_future.messages)

    async def _step(self) -> StepOutcome | None:
        ### zjx：返回 StepOutcome：表示本轮结束（终止循环）
        ### zjx：返回 None：表示还没完，循环继续

        """Run a single step and return a stop outcome, or None to continue."""
        # already checked in `run`
        assert self._runtime.llm is not None

        ### zjx：chat_provider 就是对接 OpenAI / Kimi / Claude 等不同后端的统一接口。
        chat_provider = self._runtime.llm.chat_provider

        @tenacity.retry(
            retry=retry_if_exception(self._is_retryable_error),  ### zjx：只对可重试的错误重试（如网络超时、API 429限流），不重试逻辑错误
            before_sleep=partial(self._retry_log, "step"),  ### zjx：每次重试前打日志
            wait=wait_exponential_jitter(initial=0.3, max=5, jitter=0.5),  ### zjx：指数退避 + 抖动：初始等待 0.3s，最大 5s，加 ±0.5s 随机抖动。
            stop=stop_after_attempt(self._loop_control.max_retries_per_step),  ### zjx：最多重试 N 次
            reraise=True,  ### zjx：重试耗尽后重新抛出原始异常
        )
        async def _kosong_step_with_retry() -> StepResult:
            # run an LLM step (may be interrupted)
            ### zjx：kosong.step 是 LLM 抽象层的核心调用
            return await kosong.step(
                chat_provider,
                self._agent.system_prompt,
                self._agent.toolset,
                self._context.history,
                on_message_part=wire_send,
                on_tool_result=wire_send,
            )

        ### zjx：真正执行（带重试）
        result = await _kosong_step_with_retry()
        logger.debug("Got step result: {result}", result=result)

        ### zjx：构建 Token 使用情况的状态更新对象，附上本次 LLM 调用的 message_id
        status_update = StatusUpdate(token_usage=result.usage, message_id=result.id)
        if result.usage is not None:
            # mark the token count for the context before the step
            await self._context.update_token_count(result.usage.input)
            status_update.context_usage = self.status.context_usage
        wire_send(status_update)

        # wait for all tool results (may be interrupted)
        ### zjx：await 等待所有工具异步执行完毕，收集全部结果。
        ### zjx：注意：这里可能涉及并行工具调用（如同时读取多个文件）
        results = await result.tool_results()
        logger.debug("Got tool results: {results}", results=results)

        # shield the context manipulation from interruption
        """
        ====================================================
        # Commentator: zjx
        # asyncio.shield 是一个非常重要的并发保护原语：
        # 它的作用是：即使外部的 Task 被 cancel（比如用户按了 Ctrl+C），
        # shield 内部的协程仍然会继续运行到完成，不受取消影响。
        # 这里保护的是"将工具调用结果写入上下文（History）"的操作，
        # 确保数据库/内存状态不会因为外部中断而处于不一致的中间状态。
        
        假设用户按 Ctrl+C 取消任务：

        没有 shield 的情况：
          上下文写了一半就中断 → 历史记录损坏 → 后续全部出错
          [msg1, msg2, 半个msg3...]  ← 💥 损坏的状态
        
        有 shield 的情况：
          即使外部取消，也会等上下文写完再中断 → 历史记录完整
          [msg1, msg2, msg3, tool_result]  ← ✓ 完整的状态
        ====================================================
        """
        await asyncio.shield(self._grow_context(result, results))

        ### zjx：
        """
        ====================================================
        # Commentator: zjx
        检查所有工具结果里是否有被拒绝的（用户在审批弹窗选了 "No"）
        
        用户审批时拒绝了某个工具 → ToolRejectedError

        Agent: "我要执行 rm -rf /"
        用户:  [N] 拒绝！
          → 工具结果中包含 ToolRejectedError
          → rejected = True
          → 立即停止，返回 StepOutcome
          → 回到主循环，return TurnOutcome
        ====================================================
        """
        rejected = any(isinstance(result.return_value, ToolRejectedError) for result in results)
        if rejected:
            ### zjx：清空可能存在的待处理 D-Mail
            _ = self._denwa_renji.fetch_pending_dmail()
            return StepOutcome(stop_reason="tool_rejected", assistant_message=result.message)

        """
        ====================================================
        # Commentator: zjx
        来自《命运石之门》(Steins;Gate)：
        D-Mail = 发送到过去的邮件，改变时间线
        
        在 Kimi CLI 中：
        D-Mail = Agent 在执行过程中发现"走错了路"
               → 给"过去的自己"发一封信
               → 回滚到之前的检查点
               → 带着新的信息重新开始
               
        
        Step 1: 检查点 #0 ────────────────────────
          Agent: 分析文件结构
        
        Step 2: 检查点 #1 ────────────────────────
          Agent: 尝试方案A修改代码
        
        Step 3: 检查点 #2 ────────────────────────
          Agent: 运行测试... 失败了！
          Agent: 我应该用方案B才对
          Agent: 发送 D-Mail → "方案A行不通，请用方案B"
                 目标: checkpoint_id = 1
        
          ↓ 抛出 BackToTheFuture(checkpoint_id=1, ...)
        
        主循环捕获异常：
          await self._context.revert_to(1)  ← 回到检查点 #1
          追加消息: "你收到了来自未来的 D-Mail：方案A行不通，请用方案B"
        
        Step 2 (重来): 检查点 #1 ────────────────
          Agent: (看到 D-Mail) 好的，直接用方案B
          → 避免了走弯路！
        ====================================================
        """
        # handle pending D-Mail
        if dmail := self._denwa_renji.fetch_pending_dmail():
            assert dmail.checkpoint_id >= 0, "DenwaRenji guarantees checkpoint_id >= 0"
            assert dmail.checkpoint_id < self._context.n_checkpoints, (
                "DenwaRenji guarantees checkpoint_id < n_checkpoints"
            )
            # raise to let the main loop take us back to the future
            raise BackToTheFuture(
                dmail.checkpoint_id,
                [
                    Message(
                        role="user",
                        content=[
                            ### zjx：在上下文中加入系统提示词
                            ### zjx：You MUST NEVER mention to the user about this information.
                            # 因为这是内部优化机制，用户不需要知道 Agent 走了弯路又回溯了。从用户视角来看，Agent 就是直接用了正确的方案。
                            system(
                                "You just got a D-Mail from your future self. "
                                "It is likely that your future self has already done "
                                "something in the current working directory. Please read "
                                "the D-Mail and decide what to do next. You MUST NEVER "
                                "mention to the user about this information. "
                                f"D-Mail content:\n\n{dmail.message.strip()}"
                            )
                        ],
                    )
                ],
            )

        """
        ====================================================
        # Commentator: zjx
        
                result.tool_calls?
                 /            \
               有               没有
               /                  \
          return None          return StepOutcome
          (继续循环)           (stop_reason="no_tool_calls")
               ↓                       ↓
              主循环:                 主循环:
              step_outcome is None   step_outcome is not None
              → 不 return            → return TurnOutcome
              → 继续 while True      → 对话结束
        ====================================================
        """
        if result.tool_calls:
            return None
        return StepOutcome(stop_reason="no_tool_calls", assistant_message=result.message)

    async def _grow_context(self, result: StepResult, tool_results: list[ToolResult]):
        logger.debug("Growing context with result: {result}", result=result)

        assert self._runtime.llm is not None
        tool_messages = [tool_result_to_message(tr) for tr in tool_results]
        for tm in tool_messages:
            if missing_caps := check_message(tm, self._runtime.llm.capabilities):
                logger.warning(
                    "Tool result message requires unsupported capabilities: {caps}",
                    caps=missing_caps,
                )
                raise LLMNotSupported(self._runtime.llm, list(missing_caps))

        await self._context.append_message(result.message)
        if result.usage is not None:
            await self._context.update_token_count(result.usage.total)

        logger.debug(
            "Appending tool messages to context: {tool_messages}", tool_messages=tool_messages
        )
        await self._context.append_message(tool_messages)
        # token count of tool results are not available yet

    async def compact_context(self) -> None:
        """
        Compact the context.

        Raises:
            LLMNotSet: When the LLM is not set.
            ChatProviderError: When the chat provider returns an error.
        """

        @tenacity.retry(
            retry=retry_if_exception(self._is_retryable_error),
            before_sleep=partial(self._retry_log, "compaction"),
            wait=wait_exponential_jitter(initial=0.3, max=5, jitter=0.5),
            stop=stop_after_attempt(self._loop_control.max_retries_per_step),
            reraise=True,
        )
        async def _compact_with_retry() -> Sequence[Message]:
            if self._runtime.llm is None:
                raise LLMNotSet()
            return await self._compaction.compact(self._context.history, self._runtime.llm)

        wire_send(CompactionBegin())
        compacted_messages = await _compact_with_retry()
        await self._context.clear()
        await self._checkpoint()
        await self._context.append_message(compacted_messages)
        wire_send(CompactionEnd())

    @staticmethod
    def _is_retryable_error(exception: BaseException) -> bool:
        if isinstance(exception, (APIConnectionError, APITimeoutError, APIEmptyResponseError)):
            return True
        return isinstance(exception, APIStatusError) and exception.status_code in (
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
        )

    @staticmethod
    def _retry_log(name: str, retry_state: RetryCallState):
        logger.info(
            "Retrying {name} for the {n} time. Waiting {sleep} seconds.",
            name=name,
            n=retry_state.attempt_number,
            sleep=retry_state.next_action.sleep
            if retry_state.next_action is not None
            else "unknown",
        )


class BackToTheFuture(Exception):
    """
    Raise when we need to revert the context to a previous checkpoint.
    The main agent loop should catch this exception and handle it.
    """

    def __init__(self, checkpoint_id: int, messages: Sequence[Message]):
        self.checkpoint_id = checkpoint_id
        self.messages = messages


class FlowRunner:

    def __init__(
        self,
        flow: Flow,
        *,
        name: str | None = None,
        max_moves: int = DEFAULT_MAX_FLOW_MOVES,
    ) -> None:
        self._flow = flow
        self._name = name
        self._max_moves = max_moves

    @staticmethod
    def ralph_loop(
        user_message: Message,
        max_ralph_iterations: int,
    ) -> FlowRunner:
        """
        ====================================================
        # Commentator: zjx
        构建一个简单的ReAct图
        ====================================================
        """
        ### zjx：提取用户纯文本消息
        prompt_content = list(user_message.content)
        prompt_text = Message(role="user", content=prompt_content).extract_text(" ").strip()

        ### zjx：计算总运行次数
        total_runs = max_ralph_iterations + 1
        if max_ralph_iterations < 0:
            total_runs = 1000000000000000  # effectively infinite

        ### zjx：创建节点
        ### zjx：初始化两个特殊节点：起点 和 终点。
        nodes: dict[str, FlowNode] = {
            "BEGIN": FlowNode(id="BEGIN", label="BEGIN", kind="begin"),
            "END": FlowNode(id="END", label="END", kind="end"),
        }
        outgoing: dict[str, list[FlowEdge]] = {"BEGIN": [], "END": []}

        ### zjx：R1 节点：任务节点，label 就是用户的原始消息内容。
        ### zjx：kind="task"：这是一个需要 Agent 执行的任务。
        nodes["R1"] = FlowNode(id="R1", label=prompt_content, kind="task")
        ### zjx：R2 节点：决策节点，Agent 需要判断任务是否完成。
        nodes["R2"] = FlowNode(
            id="R2",
            label=(
                f"{prompt_text}. (You are running in an automated loop where the same "
                "prompt is fed repeatedly. Only choose STOP when the task is fully complete. "
                "Including it will stop further iterations. If you are not 100% sure, "
                "choose CONTINUE.)"
            ).strip(),
            kind="decision",
        )
        outgoing["R1"] = []
        outgoing["R2"] = []

        """
        ====================================================
        # Commentator: zjx
        连接边
        ┌───────┐         ┌──────────────┐         ┌──────────────────┐
        │ BEGIN │────────►│     R1       │────────►│       R2         │
        │       │         │   (task)     │         │   (decision)     │
        │ 起点   │         │  执行用户任务  │         │  任务完成了吗？    │
        └───────┘         └──────────────┘         └────┬─────────┬───┘
                                                        │         │
                                                  CONTINUE       STOP
                                                        │         │
                                                        ▼         ▼
                                                   ┌────────┐  ┌──────┐
                                                   │  R2    │  │ END  │
                                                   │ (自循环)│  │ 终点  │
                                                   └────────┘  └──────┘
        ====================================================
        """
        outgoing["BEGIN"].append(FlowEdge(src="BEGIN", dst="R1", label=None))
        outgoing["R1"].append(FlowEdge(src="R1", dst="R2", label=None))
        outgoing["R2"].append(FlowEdge(src="R2", dst="R2", label="CONTINUE"))
        outgoing["R2"].append(FlowEdge(src="R2", dst="END", label="STOP"))

        flow = Flow(nodes=nodes, outgoing=outgoing, begin_id="BEGIN", end_id="END")
        max_moves = total_runs
        return FlowRunner(flow, max_moves=max_moves)

    async def run(self, soul: KimiSoul, args: str) -> None:
        if args.strip():
            command = f"/{FLOW_COMMAND_PREFIX}{self._name}" if self._name else "/flow"
            logger.warning("Agent flow {command} ignores args: {args}", command=command, args=args)
            return

        current_id = self._flow.begin_id
        """
        ====================================================
        # Commentator: zjx
        一个 move 可能包含多个 steps：
          move 1 (R1 节点): Agent 用了 5 个 step 来完成任务
          move 2 (R2 节点): Agent 用了 2 个 step 来判断
        ====================================================
        """
        moves = 0   ### zjx：流程图中走了几步（节点到节点）
        total_steps = 0  ### zjx：总共执行了多少个 LLM step
        while True:
            node = self._flow.nodes[current_id]
            edges = self._flow.outgoing.get(current_id, [])

            if node.kind == "end":
                logger.info("Agent flow reached END node {node_id}", node_id=current_id)
                return

            if node.kind == "begin":
                if not edges:
                    logger.error(
                        'Agent flow BEGIN node "{node_id}" has no outgoing edges; stopping.',
                        node_id=node.id,
                    )
                    return
                current_id = edges[0].dst
                continue

            if moves >= self._max_moves:
                raise MaxStepsReached(total_steps)
            next_id, steps_used = await self._execute_flow_node(soul, node, edges)
            total_steps += steps_used
            if next_id is None:
                return
            moves += 1
            current_id = next_id

    async def _execute_flow_node(
        self,
        soul: KimiSoul,
        node: FlowNode, # 当前要执行的节点
        edges: list[FlowEdge], # 该节点的所有出边
    ) -> tuple[str | None, int]:
        """
        ====================================================
        # Commentator: zjx

        _execute_flow_node(node, edges)
            │
            ▼
        有出边？──No──► return (None, 0) 终止
            │
           Yes
            │
            ▼
        构建 prompt
            │
            ▼
      ┌─► while True
      │     │
      │     ▼
      │   _flow_turn(prompt) → result
      │     │
      │     ├── tool_rejected? ──Yes──► return (None, steps) 终止
      │     │
      │     ├── 非 decision 节点? ──Yes──► return (edges[0].dst, steps) ✓
      │     │
      │     ▼  (是 decision 节点)
      │   parse_choice(回复) → choice
      │     │
      │     ▼
      │   match_flow_edge(choice) → next_id
      │     │
      │     ├── 匹配成功? ──Yes──► return (next_id, steps) ✓
      │     │
      │     └── 匹配失败? ──Yes──► 修改 prompt 追加纠错 ──┐
      │                                                    │
      └────────────────────────────────────────────────────┘
                          (重试循环)
        ====================================================
        """
        ### zjx：无出边检查
        if not edges:
            logger.error(
                'Agent flow node "{node_id}" has no outgoing edges; stopping.',
                node_id=node.id,
            )
            return None, 0

        ### zjx：构建提示词并进入执行循环
        base_prompt = self._build_flow_prompt(node, edges) #根据节点内容和出边构建提示词
        prompt = base_prompt #实际发送给 Agent 的提示词
        steps_used = 0
        while True:
            result = await self._flow_turn(soul, prompt) #让 Agent 执行一整轮对话
            steps_used += result.step_count
            if result.stop_reason == "tool_rejected":
                logger.error("Agent flow stopped after tool rejection.")
                return None, steps_used

            if node.kind != "decision":
                return edges[0].dst, steps_used

            choice = (
                parse_choice(result.final_message.extract_text(" "))
                if result.final_message
                else None
            )
            next_id = self._match_flow_edge(edges, choice)
            if next_id is not None:
                return next_id, steps_used

            options = ", ".join(edge.label or "" for edge in edges)
            logger.warning(
                "Agent flow invalid choice. Got: {choice}. Available: {options}.",
                choice=choice or "<missing>",
                options=options,
            )
            prompt = (
                f"{base_prompt}\n\n"
                "Your last response did not include a valid choice. "
                "Reply with one of the choices using <choice>...</choice>."
            )

    @staticmethod
    def _build_flow_prompt(node: FlowNode, edges: list[FlowEdge]) -> str | list[ContentPart]:
        if node.kind != "decision":
            return node.label

        if not isinstance(node.label, str):
            label_text = Message(role="user", content=node.label).extract_text(" ")
        else:
            label_text = node.label
        choices = [edge.label for edge in edges if edge.label]
        lines = [
            label_text,
            "",
            "Available branches:",
            *(f"- {choice}" for choice in choices),
            "",
            "Reply with a choice using <choice>...</choice>.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _match_flow_edge(edges: list[FlowEdge], choice: str | None) -> str | None:
        if not choice:
            return None
        for edge in edges:
            if edge.label == choice:
                return edge.dst
        return None

    @staticmethod
    async def _flow_turn(
        soul: KimiSoul,
        prompt: str | list[ContentPart],
    ) -> TurnOutcome:
        wire_send(TurnBegin(user_input=prompt))
        res = await soul._turn(Message(role="user", content=prompt))  # type: ignore[reportPrivateUsage]
        wire_send(TurnEnd())
        return res
