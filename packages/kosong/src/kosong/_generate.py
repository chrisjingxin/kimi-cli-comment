from collections.abc import Sequence
from dataclasses import dataclass

from loguru import logger

from kosong.chat_provider import (
    APIEmptyResponseError,
    ChatProvider,
    StreamedMessagePart,
    TokenUsage,
)
from kosong.message import ContentPart, Message, ToolCall
from kosong.tooling import Tool
from kosong.utils.aio import Callback, callback


async def generate(
    chat_provider: ChatProvider,
    system_prompt: str,
    tools: Sequence[Tool],
    history: Sequence[Message],
    *,
    on_message_part: Callback[[StreamedMessagePart], None] | None = None,
    on_tool_call: Callback[[ToolCall], None] | None = None,
) -> "GenerateResult":
    """
    Generate one message based on the given context.
    Parts of the message will be streamed to the specified callbacks if provided.

    Args:
        chat_provider: The chat provider to use for generation.
        system_prompt: The system prompt to use for generation.
        tools: The tools available for the model to call.
        history: The message history to use for generation.
        on_message_part: An optional callback to be called for each raw message part.
        on_tool_call: An optional callback to be called for each complete tool call.

    Returns:
        A tuple of the generated message and the token usage (if available).
        All parts in the message are guaranteed to be complete and merged as much as possible.

    Raises:
        APIConnectionError: If the API connection fails.
        APITimeoutError: If the API request times out.
        APIStatusError: If the API returns a status code of 4xx or 5xx.
        APIEmptyResponseError: If the API returns an empty response.
        ChatProviderError: If any other recognized chat provider error occurs.
    """

    """
    ====================================================
    # Commentator: zjx
    
        流式输入:  Text("你") Text("好") Text("!") ToolCall(部分1) ToolCall(部分2)
        
        处理过程:
        
        Step 1: pending=None
                → pending = Text("你")
        
        Step 2: pending=Text("你"), new=Text("好")
                → merge成功! pending = Text("你好")
        
        Step 3: pending=Text("你好"), new=Text("!")
                → merge成功! pending = Text("你好!")
        
        Step 4: pending=Text("你好!"), new=ToolCall(部分1)
                → merge失败! Text 和 ToolCall 不能合并
                → message.append(Text("你好!"))  ← 文字完成
                → pending = ToolCall(部分1)
        
        Step 5: pending=ToolCall(部分1), new=ToolCall(部分2)
                → merge成功! pending = ToolCall(完整)
        
        最终 message.content = [Text("你好!"), ToolCall(完整)]
    ====================================================
    """


    message = Message(role="assistant", content=[])
    pending_part: StreamedMessagePart | None = None  # message part that is currently incomplete

    logger.trace("Generating with history: {history}", history=history)

    ### zjx：调用 LLM API，返回一个异步流（async iterator）
    ### zjx：await 等待连接建立、请求发送
    ### zjx：此时 LLM 还没有开始返回内容，只是建立了流式连接
    stream = await chat_provider.generate(system_prompt, tools, history)
    ### zjx：异步遍历流中的每个片段
    async for part in stream:
        logger.trace("Received part: {part}", part=part)
        if on_message_part:
            ### zjx：每个碎片都立刻转发给 on_message_part，就是返回给用户终端
            await callback(on_message_part, part.model_copy(deep=True))

        if pending_part is None:
            ### zjx：第一个片段：直接作为 pending_part
            pending_part = part
        ### zjx：如果不是第一个片段，尝试合并
        elif not pending_part.merge_in_place(part):  # try merge into the pending part
            # unmergeable part must push the pending part to the buffer
            ### zjx：如果不能合并，比如类型不同，TextPart和ToolCall不能合并

            _message_append(message, pending_part) ### zjx：把攒好的文字推入 message
            if isinstance(pending_part, ToolCall) and on_tool_call:
                await callback(on_tool_call, pending_part)  ### zjx：如果 pending_part 是 ToolCall → 触发 on_tool_call 回调
            pending_part = part ### zjx：用新片段开始攒下一段

    # end of message 流结束
    if pending_part is not None: ### zjx：流结束后，最后一个 pending_part 还没有被推入 message
        _message_append(message, pending_part)
        if isinstance(pending_part, ToolCall) and on_tool_call:
            await callback(on_tool_call, pending_part)

    ### zjx：如果 LLM 什么都没返回（没有文字也没有工具调用），抛出异常
    ### zjx：这会被上层的 tenacity.retry 捕获并重试
    if not message.content and not message.tool_calls:
        raise APIEmptyResponseError("The API returned an empty response.")

    ### zjx：返回结果
    return GenerateResult(
        id=stream.id,
        message=message,
        usage=stream.usage,
    )


@dataclass(frozen=True, slots=True)
class GenerateResult:
    """The result of a generation."""

    id: str | None
    """The ID of the generated message."""
    message: Message
    """The generated message."""
    usage: TokenUsage | None
    """The token usage of the generated message."""


def _message_append(message: Message, part: StreamedMessagePart) -> None:
    match part:
        case ContentPart():
            message.content.append(part)
        case ToolCall():
            if message.tool_calls is None:
                message.tool_calls = []
            message.tool_calls.append(part)
        case _:
            # may be an orphaned `ToolCallPart`
            return
