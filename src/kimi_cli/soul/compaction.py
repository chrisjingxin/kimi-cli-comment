from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import kosong
from kosong.message import Message
from kosong.tooling.empty import EmptyToolset

import kimi_cli.prompts as prompts
from kimi_cli.llm import LLM
from kimi_cli.soul.message import system
from kimi_cli.utils.logging import logger
from kimi_cli.wire.types import ContentPart, TextPart, ThinkPart


@runtime_checkable
class Compaction(Protocol):
    async def compact(self, messages: Sequence[Message], llm: LLM) -> Sequence[Message]:
        """
        Compact a sequence of messages into a new sequence of messages.

        Args:
            messages (Sequence[Message]): The messages to compact.
            llm (LLM): The LLM to use for compaction.

        Returns:
            Sequence[Message]: The compacted messages.

        Raises:
            ChatProviderError: When the chat provider returns an error.
        """
        ...


if TYPE_CHECKING:

    def type_check(simple: SimpleCompaction):
        _: Compaction = simple


class SimpleCompaction:
    def __init__(self, max_preserved_messages: int = 2) -> None:
        self.max_preserved_messages = max_preserved_messages ### zjx：保留最近几条消息不被压缩（默认 2 条）

    async def compact(self, messages: Sequence[Message], llm: LLM) -> Sequence[Message]:
        compact_message, to_preserve = self.prepare(messages)
        if compact_message is None:
            return to_preserve

        # Call kosong.step to get the compacted context
        # TODO: set max completion tokens
        ### zjx：调用 LLM 进行压缩
        logger.debug("Compacting context...")
        result = await kosong.step(
            chat_provider=llm.chat_provider,
            system_prompt="You are a helpful assistant that compacts conversation context.",
            toolset=EmptyToolset(),
            history=[compact_message],
        )
        ### zjx：记录压缩操作本身消耗的 token
        if result.usage:
            logger.debug(
                "Compaction used {input} input tokens and {output} output tokens",
                input=result.usage.input,
                output=result.usage.output,
            )
        ### zjx：创建一条特殊的消息，开头带一个 system 标记，告诉 Agent：之前的上下文已被压缩，以下是压缩后的内容
        content: list[ContentPart] = [
            system("Previous context has been compacted. Here is the compaction output:")
        ]
        compacted_msg = result.message

        # drop thinking parts if any
        ### zjx：过滤思考过程
        content.extend(part for part in compacted_msg.content if not isinstance(part, ThinkPart))
        compacted_messages: list[Message] = [Message(role="user", content=content)]
        compacted_messages.extend(to_preserve)
        """
        ====================================================
        # Commentator: zjx
        [
            Message(role="user", content=[
                SystemPart("Previous context has been compacted..."),
                TextPart("摘要：用户最初请求帮助修复项目中的 lint 错误。
                          Agent 运行了 eslint，发现 5 个错误。
                          已修复了 a.js 和 b.js 中的 3 个错误。
                          用户随后要求也修复 TypeScript 错误..."),
            ]),
            
            Message(role="assistant", content=[...]),    ← 保留的最近消息
            Message(role="user", content=[...]),          ← 保留的最近消息
        ]
        ====================================================
        """
        return compacted_messages


    class PrepareResult(NamedTuple):
        compact_message: Message | None
        to_preserve: Sequence[Message]

    def prepare(self, messages: Sequence[Message]) -> PrepareResult:
        ### zjx：准备压缩的消息
        if not messages or self.max_preserved_messages <= 0:
            return self.PrepareResult(compact_message=None, to_preserve=messages)

        history = list(messages)
        preserve_start_index = len(history)
        n_preserved = 0
        for index in range(len(history) - 1, -1, -1):
            if history[index].role in {"user", "assistant"}:
                n_preserved += 1
                if n_preserved == self.max_preserved_messages:
                    preserve_start_index = index
                    break

        if n_preserved < self.max_preserved_messages:
            return self.PrepareResult(compact_message=None, to_preserve=messages)

        to_compact = history[:preserve_start_index]
        to_preserve = history[preserve_start_index:]

        if not to_compact:
            # Let's hope this won't exceed the context size limit
            return self.PrepareResult(compact_message=None, to_preserve=to_preserve)

        # Create input message for compaction
        compact_message = Message(role="user", content=[])
        for i, msg in enumerate(to_compact):
            compact_message.content.append(
                TextPart(text=f"## Message {i + 1}\nRole: {msg.role}\nContent:\n")
            )
            compact_message.content.extend(
                part for part in msg.content if not isinstance(part, ThinkPart)
            )
        compact_message.content.append(TextPart(text="\n" + prompts.COMPACT))
        return self.PrepareResult(compact_message=compact_message, to_preserve=to_preserve)
