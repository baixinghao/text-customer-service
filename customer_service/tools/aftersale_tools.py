"""售后域工具：退款申请落库 refund_applications。

两条纪律：
- 手机号/金额不信任 LLM 传入，按 order_id 从订单快照查出落库（快照冗余的意义）
- 幂等键是业务约束（同一订单同时只能有一张"进行中"申请），不是日历约束；
  DB 层有 partial unique index 兜底并发，代码里先 SELECT 给友好提示

（退换货政策检索已迁去 rag/retriever.py 的 get_policy_retriever_tool，
  本文件不再维护政策文本。）
"""

import secrets
from datetime import datetime

import psycopg2.errors
from langchain.tools import tool

from customer_service.db import get_conn
from customer_service.tools.order_tools import STATUS_ZH, get_order

# 进行中的申请状态：同一订单同时只允许一张。
# 与 refund_applications 上的 partial unique index
# （WHERE status IN ('待审核','审核通过')）保持一致，改这里要一起改 DDL。
_IN_FLIGHT_STATUSES = ("待审核", "审核通过")


def _gen_request_no() -> str:
    """退款申请单号：RF + 时间戳 + 随机位，不可被外部推算。"""
    return f"RF{datetime.now():%Y%m%d%H%M%S}{secrets.token_hex(3).upper()}"


@tool
def apply_refund(order_id: str, reason: str) -> str:
    """为指定订单提交退款申请。order_id 为订单号，reason 为退款原因。
    退款金额由系统按 order_id 从订单表查出做快照，不接受外部传入；
    联系电话库里暂无来源（app_user 无 phone 字段），留空待接入真实用户体系。"""
    order = get_order(order_id)
    if not order:
        return f"退款失败：未找到订单 {order_id}，请确认订单号是否正确"
    if order["status"] == "created":
        return f"订单 {order['id']} 尚未付款，无需退款，可直接取消订单"
    if order["status"] in ("closed", "refunded"):
        return f"订单 {order['id']} 当前状态为【{STATUS_ZH[order['status']]}】，不能申请退款"

    oid = str(order["id"])  # 与 refund_applications.order_id 的存储格式保持一致
    with get_conn() as conn:
        with conn.cursor() as cur:
            # 幂等：同一订单同时只能有一张"进行中"的申请。
            # 不是"一天一张"——被拒后明天可以再申请；也不限一天——
            # 跨天重复发起，只要上一张还在进行中就要拦。
            placeholders = ", ".join(["%s"] * len(_IN_FLIGHT_STATUSES))
            cur.execute(
                f"""SELECT request_no, status FROM refund_applications
                    WHERE order_id = %s AND status IN ({placeholders})
                    ORDER BY id DESC LIMIT 1""",
                (oid, *_IN_FLIGHT_STATUSES),
            )
            row = cur.fetchone()
            if row:
                return (f"订单 {oid} 已有进行中的退款申请（单号 {row[0]}，"
                        f"当前状态：{row[1]}），无需重复提交")

            request_no = _gen_request_no()
            try:
                cur.execute(
                    """INSERT INTO refund_applications
                         (request_no, order_id, phone, amount, reason)
                       VALUES (%s, %s, NULL, %s, %s)""",
                    # 金额快照来自订单表，不是 LLM 参数；phone 库内无来源，显式 NULL
                    (request_no, oid, order["amount"], reason),
                )
            except psycopg2.errors.UniqueViolation:
                # 并发兜底：同时插入两张"进行中"申请时 partial unique index 拦截。
                # 事务已中止，显式回滚后再降级成友好提示
                conn.rollback()
                return f"订单 {oid} 刚刚已提交过退款申请，请稍后再试"
    return (f"已为订单 {oid}（{order['items']}）提交退款申请，单号 {request_no}，"
            f"当前状态：待审核。审核通过后 1-3 个工作日原路到账")
