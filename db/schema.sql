-- 真实库表结构（存储红线：一律走 PostgreSQL）
-- 用法：PYTHONIOENCODING=utf-8 .venv/Scripts/python db/apply_schema.py
-- 设计原则：Mock（tools/ 顶部常量）按计划逐步换成这里的真表，换一张删一段 Mock

-- ==========================================================================
-- 退款申请表：AI 只"发起"，审批归人/规则引擎（纪律：LLM 可以发起，不能决定）
-- 状态机：待审核 → 审核通过 / 已拒绝 →（审核通过后）已到账
-- ==========================================================================
CREATE TABLE IF NOT EXISTS refund_applications (
    id          BIGSERIAL PRIMARY KEY,
    request_no  TEXT NOT NULL UNIQUE,              -- 申请单号，代码生成（TK + 时间戳 + 随机位）
    order_id    TEXT NOT NULL,                     -- 订单号（现在对 MOCK_ORDERS，将来对订单真表）
    phone       TEXT NOT NULL,                     -- 申请人手机号，从订单冗余，便于按人查询
    amount      NUMERIC(10, 2),                    -- 申请金额，从订单冗余（订单将来会改，快照留证）
    reason      TEXT NOT NULL,                     -- 退款原因（LLM 收集，工具代码负责校验非空/长度）
    status      TEXT NOT NULL DEFAULT '待审核',     -- 状态机见上，代码里枚举校验，不许自由文本
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_refund_applications_order ON refund_applications (order_id);
CREATE INDEX IF NOT EXISTS idx_refund_applications_phone ON refund_applications (phone);

-- 幂等兜底：同一订单同时只能有一张"进行中"的申请（代码里 SELECT 先拦给友好提示，
-- 这个 partial unique index 拦并发竞态）。与 aftersale_tools._IN_FLIGHT_STATUSES 保持一致
CREATE UNIQUE INDEX IF NOT EXISTS uq_refund_in_flight
    ON refund_applications (order_id)
    WHERE status IN ('待审核', '审核通过');

-- ==========================================================================
-- P4 工单：tickets + ticket_events（精简版，学习用）
-- 完整设计分析（两层幂等/信号vs噪声）见 intent-gate 产物，此处只留落地版
--
-- 幂等语义（与退款申请对照，范式相同、语义相反）：
--   退款重复提交 = 无信息量 → 拦掉；工单重复投诉 = 有信息量（没解决/情绪升级）
--   → 不拦不新建，并入既有进行中工单：content 原文进 ticket_events，
--     escalation_count +1。连骂三次 = 一张单、三条事件、计数 3
-- ==========================================================================
CREATE TABLE IF NOT EXISTS tickets (
    id               BIGSERIAL PRIMARY KEY,
    ticket_no        TEXT        NOT NULL UNIQUE,  -- GD+时间戳+随机位（同 RF 范式，可枚举单号是泄露面）
    phone            TEXT        NOT NULL,
    category         TEXT        NOT NULL,          -- 开放枚举（投诉/退款/技术问题…），不加 CHECK
    content          TEXT        NOT NULL,          -- 首次诉求原文，后续重复投诉不覆盖（进 events）
    priority         TEXT        NOT NULL DEFAULT '中',
    status           TEXT        NOT NULL DEFAULT '待处理',
    escalation_count INTEGER     NOT NULL DEFAULT 1, -- 重复投诉计数；>=2 时代码自动提优先级
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- 状态机是业务契约，DB 层拒绝非法状态（防代码漏改写出脏状态、保下面索引的 WHERE 语义）
    CONSTRAINT ck_tickets_status CHECK (status IN ('待处理', '处理中', '已解决', '已关闭'))
    -- 知识点（表外的账）：priority 中文枚举按 Unicode 排序是"中<低<高"——恰好是错的。
    -- 真做工作台排序时要加 priority_rank 生成列；练习里没有队列页，不建
);

-- 业务层幂等并发兜底：同人同类同时只有一张"进行中"主单（已结案的不挡新问题）
CREATE UNIQUE INDEX IF NOT EXISTS uq_tickets_in_flight
    ON tickets (phone, category)
    WHERE status IN ('待处理', '处理中');

CREATE INDEX IF NOT EXISTS idx_tickets_phone ON tickets (phone, created_at DESC);

-- 事件流水：合并语义的落点。用户重复投诉的原文一字不丢，全在时间线上
CREATE TABLE IF NOT EXISTS ticket_events (
    id         BIGSERIAL   PRIMARY KEY,
    ticket_no  TEXT        NOT NULL REFERENCES tickets (ticket_no) ON DELETE CASCADE,
    event_type TEXT        NOT NULL,  -- 建单/重复投诉/追加记录（代码枚举校验）
    content    TEXT,                  -- 重复投诉时是用户本次原话
    actor      TEXT        NOT NULL,  -- 用户 / AI / 人工 / 系统
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ticket_events_ticket ON ticket_events (ticket_no, id);
