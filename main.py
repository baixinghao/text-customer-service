"""CLI 入口：本地命令行体验多轮客服对话。

运行（Git Bash）：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python main.py
"""

from customer_service.graph import build_graph


def main():
    graph = build_graph()
    # 同一 thread_id 下的多轮对话共享 checkpointer 记忆；换个 id 就是新会话
    config = {"configurable": {"thread_id": "cli-demo-1"}}

    print("智能客服已上线，输入 quit 退出\n")
    while True:
        question = input("用户: ").strip()
        if not question or question.lower() in ("quit", "exit"):
            break

        result = graph.invoke({"messages": [("user", question)]}, config)
        print(f"客服[{result.get('active_agent', '?')}]: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
