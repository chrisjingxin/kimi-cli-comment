Spawn a subagent to perform a specific task. Subagent will be spawned with a fresh context without any history of yours.

**Context Isolation**

Context isolation is one of the key benefits of using subagents. By delegating tasks to subagents, you can keep your main context clean and focused on the main goal requested by the user.

Here are some scenarios you may want this tool for context isolation:

- You wrote some code and it did not work as expected. In this case you can spawn a subagent to fix the code, asking the subagent to return how it is fixed. This can potentially benefit because the detailed process of fixing the code may not be relevant to your main goal, and may clutter your context.
- When you need some latest knowledge of a specific library, framework or technology to proceed with your task, you can spawn a subagent to search on the internet for the needed information and return to you the gathered relevant information, for example code examples, API references, etc. This can avoid ton of irrelevant search results in your own context.

DO NOT directly forward the user prompt to Task tool. DO NOT simply spawn Task tool for each todo item. This will cause the user confused because the user cannot see what the subagent do. Only you can see the response from the subagent. So, only spawn subagents for very specific and narrow tasks like fixing a compilation error, or searching for a specific solution.

**Parallel Multi-Tasking**

Parallel multi-tasking is another key benefit of this tool. When the user request involves multiple subtasks that are independent of each other, you can use Task tool multiple times in a single response to let subagents work in parallel for you.

Examples:

- User requests to code, refactor or fix multiple modules/files in a project, and they can be tested independently. In this case you can spawn multiple subagents each working on a different module/file.
- When you need to analyze a huge codebase (>hundreds of thousands of lines), you can spawn multiple subagents each exploring on a different part of the codebase and gather the summarized results.
- When you need to search the web for multiple queries, you can spawn multiple subagents for better efficiency.

**Available Subagents:**

${SUBAGENTS_MD}

---

派生一个子代理来执行特定任务。子代理将以全新的上下文启动，不包含你的任何历史记录。

**上下文隔离**

上下文隔离是使用子代理的关键优势之一。通过将任务委派给子代理，你可以保持主上下文的整洁，专注于用户请求的主要目标。

以下是一些你可能需要使用此工具进行上下文隔离的场景：

- 你写了一些代码但运行结果不符合预期。在这种情况下，你可以派生一个子代理来修复代码，并让子代理返回修复方式。这样做的好处是，修复代码的详细过程可能与你的主要目标无关，会占用你的上下文空间。
- 当你需要某个特定库、框架或技术的最新知识来推进任务时，你可以派生一个子代理在互联网上搜索所需信息，并将收集到的相关信息返回给你，例如代码示例、API 参考文档等。这可以避免大量无关的搜索结果出现在你自己的上下文中。

请勿直接将用户的提示词转发给 Task 工具。请勿简单地为每个待办事项都派生 Task 工具。这会让用户感到困惑，因为用户无法看到子代理在做什么。只有你能看到子代理的响应。因此，仅为非常具体和狭窄的任务派生子代理，例如修复编译错误或搜索特定解决方案。

**并行多任务处理**

并行多任务处理是此工具的另一个关键优势。当用户请求涉及多个相互独立的子任务时，你可以在单次响应中多次使用 Task 工具，让子代理为你并行工作。

示例：

- 用户请求对项目中的多个模块/文件进行编码、重构或修复，且它们可以独立测试。在这种情况下，你可以派生多个子代理，每个负责处理不同的模块/文件。
- 当你需要分析一个大型代码库（超过数十万行代码）时，你可以派生多个子代理，各自探索代码库的不同部分，并汇总结果。
- 当你需要在网上搜索多个查询时，你可以派生多个子代理以提高效率。

**可用的子代理：**

${SUBAGENTS_MD}
