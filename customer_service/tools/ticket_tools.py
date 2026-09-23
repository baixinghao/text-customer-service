"""工单域工具（P4）：AI 处理不了 / 用户强烈不满时，把问题沉淀成工单交人工跟进。

核心设计：合并优先于新建——同一手机号+分类同时只允许一张"进行中"工单
（DB 层 uq_tickets_in_flight partial unique index 物理兜底，与 refund 的
幂等约束同一套路）。重复进线并入原工单：原文进 ticket_events 事件流，
escalation_count+1，累计 ≥2 自动升级 priority=高。

phone 不进工具签名：由 ToolRuntime 从共享 State 读（user_phone），
来源是外层（API 网关/CLI），LLM 摸不到——和 apply_refund 的快照纪律一脉相承。
"""

import secrets
from datetime import datetime

import psycopg2.errors
from langchain.tools import tool, ToolRuntime

from customer_service.db import get_conn

# 进行中的工单状态：与 tickets.uq_tickets_in_flight 的 WHERE 子句保持一致，
# 改这里要一起改 DDL（CHECK 约束里还有 已解决/已关闭 两个终态）
_IN_FLIGHT_STATUSES = ("待处理", "处理中")


def _gen_ticket_no() -> str:
    """工单号：GD + 时间戳 + 随机位，不可被外部推算。"""
    return f"GD{datetime.now():%Y%m%d%H%M%S}{secrets.token_hex(3).upper()}"


def _load_in_flight(cur, phone: str, category: str):
    """查同手机号同分类的进行中工单，返回 (ticket_no, escalation_count) 或 None。"""
    placeholders = ", ".join(["%s"] * len(_IN_FLIGHT_STATUSES))
    cur.execute(
        f"""SELECT ticket_no, escalation_count FROM tickets
            WHERE phone = %s AND category = %s AND status IN ({placeholders})
            ORDER BY id DESC LIMIT 1""",
        (phone, category, *_IN_FLIGHT_STATUSES),
    )
    return cur.fetchone()


def _merge(cur, ticket_no: str, escalation_count: int, content: str) -> str:
    """并入已有工单：原文进事件流，计数 +1，≥2 原子升级高优先级。"""
    new_count = escalation_count + 1
    # 一条 UPDATE 原子完成 读-改-写，并发进线不会互相覆盖
    cur.execute(
        """UPDATE tickets
           SET escalation_count = %s,
               priority = CASE WHEN %s >= 2 THEN '高' ELSE priority END,
               updated_at = now()
           WHERE ticket_no = %s""",
        (new_count, new_count, ticket_no),
    )
    cur.execute(
        """INSERT INTO ticket_events (ticket_no, event_type, content, actor)
           VALUES (%s, '进线合并', %s, 'ai_agent')""",
        (ticket_no, content),
    )
    suffix = "，已自动升级高优先级" if new_count >= 2 else ""
    return (f"您已有进行中的同类工单，已并入工单 {ticket_no}"
            f"（第 {new_count} 次进线{suffix}），客服将在 24 小时内跟进")


@tool
def create_ticket(runtime: ToolRuntime, category: str, content: str, priority: str = "中") -> str:
    """创建人工跟进工单。category 如 投诉/退款/物流问题；priority 为高/中/低，
    由 AI 按情绪与严重度判断。同一手机号同分类已有进行中工单时自动并入，不重复建单。"""
    phone = (runtime.state.get("user_phone") or "").strip()
    if not phone:
        return "创建工单失败：会话中还没有您的联系电话，请先告知手机号，方便客服跟进"

    with get_conn() as conn:
        with conn.cursor() as cur:
            row = _load_in_flight(cur, phone, category)
            if row:
                return _merge(cur, row[0], row[1], content)

            ticket_no = _gen_ticket_no()
            try:
                cur.execute(
                    """INSERT INTO tickets (ticket_no, phone, category, content, priority)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (ticket_no, phone, category, content, priority),
                )
            except psycopg2.errors.UniqueViolation:
                # 并发兜底：另一请求刚建了同 phone+category 的进行中工单。
                # 事务已中止，回滚后降级走合并路径重读
                conn.rollback()
                with conn.cursor() as cur:
                    row = _load_in_flight(cur, phone, category)
                    if row is None:
                        return "创建工单失败：系统繁忙，请稍后再试"
                    return _merge(cur, row[0], row[1], content)

            # 建单成功，记一条"建单"事件，事件流从这里开始
            cur.execute(
                """INSERT INTO ticket_events (ticket_no, event_type, content, actor)
                   VALUES (%s, '建单', %s, 'ai_agent')""",
                (ticket_no, content),
            )
    return (f"已为您创建工单 {ticket_no}（{category}，优先级{priority}），"
            f"客服将在 24 小时内跟进，请留意来电")


@tool
def query_ticket(ticket_no: str) -> str:
    """根据工单号查询工单处理进度，含最近三条事件。"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT ticket_no, category, status, priority, escalation_count,
                          created_at, updated_at
                   FROM tickets WHERE ticket_no = %s""",
                (ticket_no.strip(),),
            )
            row = cur.fetchone()
            if row is None:
                return f"未找到工单 {ticket_no}，请核对单号是否正确"
            cur.execute(
                """SELECT event_type, content, actor, created_at FROM ticket_events
                   WHERE ticket_no = %s ORDER BY id DESC LIMIT 3""",
                (row[0],),
            )
            events = cur.fetchall()

    head = (f"工单 {row[0]}：分类【{row[1]}】，状态【{row[2]}】，优先级【{row[3]}】，"
            f"累计进线 {row[4]} 次\n创建：{row[5]:%Y-%m-%d %H:%M}，最近更新：{row[6]:%Y-%m-%d %H:%M}")
    if events:
        lines = [f"- [{e[0]}] {e[1] or ''}（{e[2]}，{e[3]:%m-%d %H:%M}）" for e in events]
        head += "\n最新进展：\n" + "\n".join(lines)
    return head
