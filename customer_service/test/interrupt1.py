from typing import TypedDict
import warnings

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langchain_core._api.beta_decorator import LangChainBetaWarning

# v3 streaming 是实验性协议，两条 LangChainBetaWarning 不是错误；
# 要追新版协议的行为变化时删掉这行让警告重新冒出来
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

class State(TypedDict):
    approved: bool

def approval_node(state: State):
    # 这里会暂停，返回值就是外部 resume 时传进来的值
    approved = interrupt("Do you approve this action?")
    return {"approved": approved}

graph = (
    StateGraph(State)
    .add_node("approval", approval_node)
    .add_edge(START, "approval")
    .add_edge("approval", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "thread-1"}}

# 第一次跑，跑到 interrupt() 就停
stream = graph.stream_events({"approved": False}, config=config, version="v3")
_ = stream.output
print(stream.interrupted)      # True
# 注意实际形状：list 且 Interrupt 带 id（注释里旧的元组写法已过时）
print(stream.interrupts)       # [Interrupt(value='Do you approve this action?', id='...')]

# 用人输入恢复
resumed = graph.stream_events(Command(resume=True), config=config, version="v3")
print(resumed.output)          # {'approved': True}
