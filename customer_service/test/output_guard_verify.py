"""输出护栏修复验证（真模型调用，已获丽姐许可，烧额度知情）。

复现 2026-09-24 K87 误杀事故的链路：
  轮1 "给我推荐个最好的键盘" → 本轮有工具取证，应直接过闸
  轮2 "K87 键盘多少钱？"     → 跨轮引用轮1的证据。修复前：被误杀两次；
      修复后期望：零打回直接过闸
  轮3 "K87 的 RGB 版本要加多少钱？" → 语料外事实。两种结局都算对：
      模型如实说"没查到"（零打回），或编造数字被拦——此时验证打回三件套：
      被拒稿已从历史摘除、反馈消息 id 带 do-not-render- 前缀、retry_count 递增

跑法：PYTHONIOENCODING=utf-8 .venv/Scripts/python customer_service/test/output_guard_verify.py
"""

from customer_service.graph import build_graph

THREAD_ID = "guard-verify-k87-2"


def report(tag: str, result: dict, prev_len: int, expect: str) -> None:
    msgs = result["messages"]
    new_msgs = msgs[prev_len:]
    bounces = [m for m in new_msgs if getattr(m, "type", "") == "system"]
    print(f"===== {tag} =====")
    print("期望:", expect)
    print("本轮新增消息:")
    for m in new_msgs:
        kind = getattr(m, "type", "?")
        name = getattr(m, "name", "") or ""
        mid = getattr(m, "id", "") or ""
        body = str(getattr(m, "content", ""))[:70].replace("\n", " ")
        print(f"  [{kind:7s}] {name:16s} id={mid[:26]:26s} {body}")
    print("打回反馈条数:", len(bounces),
          " id 带隐身前缀:", all((getattr(m, "id", "") or "").startswith("do-not-render-") for m in bounces))
    print("retry_count:", result.get("retry_count"))
    print()


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": THREAD_ID}}

    r1 = graph.invoke({"messages": [("user", "给我推荐个最好的键盘")]}, config)
    report("轮1：推荐键盘", r1, 1, "过闸、无打回")

    r2 = graph.invoke({"messages": [("user", "K87 键盘多少钱？")]}, config)
    report("轮2：跨轮问价（K87 事故复现）", r2, len(r1["messages"]),
           "过闸、无打回——证据来自轮1的工具结果")

    r3 = graph.invoke({"messages": [("user", "K87 的 RGB 版本要加多少钱？")]}, config)
    report("轮3：语料外事实", r3, len(r2["messages"]),
           "编造则拦（摘稿+隐身反馈）；如实说没查到也合格")


if __name__ == "__main__":
    main()
