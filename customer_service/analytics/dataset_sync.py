"""评测数据集同步：把 evaluator.EVAL_CASES 迁移进 Langfuse Dataset（一次性迁移工具）。

数据集结构约定（runner 与 UI 侧都按这个契约读写）：
    input           {"phone": "...", "turns": [{"input": "用户消息"}, ...]}
    expected_output {"turns": [{"expect_agent": "...", "expect_points": [...]}, ...]}
    metadata        {"name": 用例名, "category": 分类前缀}

幂等：以 metadata.name 判重，已存在的用例跳过不覆盖——
平台上的用例可能被人在 UI 里改过，同步脚本无权冲掉。

运行：python -m customer_service.analytics.dataset_sync
"""

from __future__ import annotations

from customer_service.analytics.evaluator import EVAL_CASES
from customer_service.analytics.langfuse_client import get_langfuse

DATASET_NAME = "cs-agent-eval"

# ── RAG 专项用例（与功能用例同一个数据集，一次跑批多维打分）─────────
# 结构向功能用例对齐：单轮 = turns 里只有一条；ground_truth 给 ragas 的
# answer_correctness / context_recall / context_precision 当参照
# in_kb=False 是库外题（≥20%），专测拒答与编造
RAG_CASES: list[dict] = [
    # ── 库内题（售后政策）──
    {"name": "kb-退款到账", "in_kb": True, "source": "refund_policy.md",
     "question": "退款审核通过后多久能到账？",
     "ground_truth": "退款原路返回，审核通过后 1-3 个工作日到账。"},
    {"name": "kb-无理由退货", "in_kb": True, "source": "refund_policy.md",
     "question": "签收后多久内可以无理由退货？",
     "ground_truth": "签收后 7 天内支持无理由退货，商品需保持完好（配件、包装齐全）。"},
    {"name": "kb-耳机拆封退货", "in_kb": True, "source": "refund_policy.md",
     "question": "耳机拆开包装了还能退吗？",
     "ground_truth": "贴身用品（耳机耳塞）拆封后不支持无理由退货，未拆封可退。"},
    {"name": "kb-大促退货时限", "in_kb": True, "source": "refund_policy.md",
     "question": "双 11 期间买的东西退货时限有变化吗？",
     "ground_truth": "大促期间签收的订单，无理由退货寄回时限由 7 天延长至 15 天。"},
    {"name": "kb-价保", "in_kb": True, "source": "refund_policy.md",
     "question": "刚买就降价了，能补差价吗？",
     "ground_truth": "签收后 30 天内官方自营降价可申请价保，差价以余额形式返还，单笔订单限 1 次。"},
    {"name": "kb-保修", "in_kb": True, "source": "refund_policy.md",
     "question": "商品保修期是多久？",
     "ground_truth": "数码类商品厂家全国联保 1 年；家具类提供 3 年结构件质保。"},
    {"name": "kb-定制商品", "in_kb": True, "source": "refund_policy.md",
     "question": "刻字定制的键盘能退吗？",
     "ground_truth": "定制类商品不支持无理由退货，质量问题可换货或维修。"},
    # ── 库内题（FAQ）──
    {"name": "kb-包邮门槛", "in_kb": True, "source": "faq.md",
     "question": "你们满多少包邮？不满怎么收运费？",
     "ground_truth": "单笔订单满 99 元包邮，不满 99 元收 8 元运费。"},
    {"name": "kb-截单时间", "in_kb": True, "source": "faq.md",
     "question": "下午五点下单今天还能发货吗？",
     "ground_truth": "工作日 16 点前付款当天发货；17 点已超过截单时间，次日发货，默认顺丰。"},
    {"name": "kb-VIP门槛", "in_kb": True, "source": "faq.md",
     "question": "怎么才能成为 VIP？有什么优惠？",
     "ground_truth": "累计消费满 2000 元自动升级 VIP，享 95 折和售后优先处理通道。"},
    {"name": "kb-专票", "in_kb": True, "source": "faq.md",
     "question": "能开增值税专用发票吗？",
     "ground_truth": "目前只支持电子普通发票，暂不支持增值税专用发票。"},
    {"name": "kb-顺丰加急", "in_kb": True, "source": "faq.md",
     "question": "顺丰能加急吗？",
     "ground_truth": "可以加 10 元升级顺丰空运，全国主要城市次日达。"},
    # ── 库内题（商品库，语义检索是卖点）──
    {"name": "kb-码字键盘", "in_kb": True, "source": "products.md",
     "question": "有没有适合长时间码字的键盘？",
     "ground_truth": "推荐机械键盘 K87（¥299）：Gasket 结构敲击软弹，静音红轴，PBT 键帽，适合长时间码字。"},
    {"name": "kb-K87vsK75", "in_kb": True, "source": "faq.md",
     "question": "K87 和 K75 有什么区别？",
     "ground_truth": "K87 ¥299：87 键、三模连接、PBT 键帽、支持热插拔；K75 ¥199：75 配列、有线单模、ABS 键帽、不支持换轴。"},
    {"name": "kb-游戏耳机", "in_kb": True, "source": "products.md",
     "question": "打游戏开黑用什么耳机好？",
     "ground_truth": "推荐头戴式游戏耳机 H7（¥549）：7.1 虚拟环绕声听声辨位，2.4G 无线低延迟，带可拆卸降噪麦克风。"},
    {"name": "kb-修图显示器", "in_kb": True, "source": "products.md",
     "question": "修图剪视频选哪款显示器？",
     "ground_truth": "选 27 寸 4K 显示器（¥1299）：4K 60Hz，99% sRGB 色域，颜色准；打游戏才选 2K 电竞款。"},
    # ── 库外题（测拒答与编造）──
    {"name": "oob-比特币", "in_kb": False,
     "question": "你们支持用比特币付款吗？", "ground_truth": ""},
    {"name": "oob-卖笔记本", "in_kb": False,
     "question": "你们店里卖笔记本电脑吗？", "ground_truth": ""},
    {"name": "oob-上门安装", "in_kb": False,
     "question": "买显示器你们提供上门安装服务吗？", "ground_truth": ""},
    {"name": "oob-积分换现金", "in_kb": False,
     "question": "会员积分怎么兑换现金？", "ground_truth": ""},
    {"name": "oob-K87防水", "in_kb": False,
     "question": "K87 键盘防水吗？掉水里还能用吗？", "ground_truth": ""},
]


def sync_dataset(dataset_name: str = DATASET_NAME) -> None:
    client = get_langfuse()
    client.create_dataset(
        name=dataset_name,
        description="客服 Agent 质检用例集（功能多轮 + RAG 专项，一个数据集多维打分）",
    )
    existing = {
        (item.metadata or {}).get("name")
        for item in client.get_dataset(dataset_name).items
    }

    created = skipped = 0
    for case in EVAL_CASES:
        if case["name"] in existing:
            skipped += 1
            continue
        client.create_dataset_item(
            dataset_name=dataset_name,
            input={
                "phone": case.get("phone", ""),
                "turns": [{"input": t["input"]} for t in case["turns"]],
            },
            expected_output={
                "turns": [
                    {
                        "expect_agent": t.get("expect_agent"),
                        "expect_points": t.get("expect_points", []),
                    }
                    for t in case["turns"]
                ]
            },
            metadata={
                "name": case["name"],
                "category": case["name"].split("-")[0],
            },
        )
        created += 1

    for case in RAG_CASES:
        if case["name"] in existing:
            skipped += 1
            continue
        client.create_dataset_item(
            dataset_name=dataset_name,
            input={
                "phone": "",
                "in_kb": case["in_kb"],
                "turns": [{"input": case["question"]}],
            },
            expected_output={
                "turns": [{"expect_agent": None, "expect_points": []}],
                "ground_truth": case["ground_truth"],
            },
            metadata={
                "name": case["name"],
                "category": "rag-in-kb" if case["in_kb"] else "rag-out-of-kb",
                "source": case.get("source", ""),
            },
        )
        created += 1

    print(f"数据集 {dataset_name}：新增 {created} 条，跳过已存在 {skipped} 条，"
          f"平台总数 {len(client.get_dataset(dataset_name).items)} 条")


if __name__ == "__main__":
    sync_dataset()
