"""订单域工具：直查 orders / order_item / product / app_user（Mock 已下线）。

orders.status 是英文枚举（CHECK 约束），对外展示统一走 STATUS_ZH 翻译。
"""

from langchain.tools import tool

from customer_service.db import get_conn

# orders.status CHECK 约束：created/paid/shipped/closed/refunded
STATUS_ZH = {
    "created": "待付款",
    "paid": "已付款",
    "shipped": "已发货",
    "closed": "已关闭",
    "refunded": "已退款",
}


def get_order(order_id: str) -> dict | None:
    """按订单号查订单 + 商品明细。

    返回 {"id", "status", "amount", "created_at", "user", "items"}；
    查不到返回 None。apply_refund 的快照也走这里——一个查询入口，别散两处。
    """
    try:
        oid = int(order_id)
    except (TypeError, ValueError):
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT o.id, o.status, o.amount, o.created_at, u.nickname,
                          COALESCE(string_agg(p.title || ' x' || oi.qty, '、'), '')
                   FROM orders o
                   JOIN app_user u ON u.id = o.user_id
                   LEFT JOIN order_item oi ON oi.order_id = o.id
                   LEFT JOIN product p ON p.id = oi.product_id
                   WHERE o.id = %s
                   GROUP BY o.id, o.status, o.amount, o.created_at, u.nickname""",
                (oid,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "status": row[1],
        "amount": float(row[2]),
        "created_at": row[3],
        "user": row[4],
        "items": row[5],
    }


@tool
def query_order(order_id: str) -> str:
    """根据订单号查询单个订单的详情：商品明细、金额、状态、下单时间。"""
    order = get_order(order_id)
    if not order:
        return f"未找到订单 {order_id}，请确认订单号是否正确"
    return (
        f"订单 {order['id']}：商品【{order['items']}】，金额 ¥{order['amount']}，"
        f"状态【{STATUS_ZH[order['status']]}】，下单时间：{order['created_at']:%Y-%m-%d %H:%M}"
    )


@tool
def list_orders_by_email(email: str) -> str:
    """根据注册邮箱列出该用户名下的订单号（含状态），用于用户不知道订单号的场景。"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT o.id, o.status FROM orders o
                   JOIN app_user u ON u.id = o.user_id
                   WHERE u.email = %s
                   ORDER BY o.created_at DESC
                   LIMIT 20""",
                (email,),
            )
            rows = cur.fetchall()
    if not rows:
        return f"邮箱 {email} 名下没有订单"
    items = "、".join(f"{oid}（{STATUS_ZH[s]}）" for oid, s in rows)
    return f"邮箱 {email} 名下的订单：{items}"
