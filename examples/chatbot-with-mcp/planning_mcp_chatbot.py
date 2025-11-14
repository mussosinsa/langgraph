"""
LangGraph와 MCP Tool을 사용한 Planning 챗봇 에이전트

이 예제는 다음 단계를 구현합니다:
1. 쿼리 입력
2. 쿼리 처리 계획 수립
3. 필요한 MCP의 Tool로 호출
4. 쿼리 처리 완료될 때까지 반복 실행 (Agent)
5. 결과 사용자에게 답변
"""
import asyncio
from typing import Annotated, TypedDict, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field


# ============================================================================
# 1. State 정의
# ============================================================================

class AgentState(TypedDict):
    """에이전트의 상태를 정의합니다."""
    # 사용자 쿼리
    query: str
    # 실행 계획
    plan: str
    # 현재까지의 대화 메시지
    messages: Annotated[list[BaseMessage], add_messages]
    # 최종 응답
    final_response: str
    # 반복 횟수 (무한 루프 방지)
    iteration: int


# ============================================================================
# 2. 계획 생성 스키마
# ============================================================================

class Plan(BaseModel):
    """쿼리 처리를 위한 실행 계획"""
    steps: list[str] = Field(
        description="쿼리를 해결하기 위한 단계별 계획 목록"
    )
    required_tools: list[str] = Field(
        description="계획 실행에 필요한 도구 목록"
    )


# ============================================================================
# 3. MCP 클라이언트 및 도구 설정
# ============================================================================

async def setup_mcp_tools():
    """MCP 서버에서 도구를 가져옵니다."""
    client = MultiServerMCPClient({
        # 계산기 MCP 서버
        "calculator": {
            "command": "python",
            "args": ["mcp_servers/calculator_server.py"],
            "transport": "stdio",
        },
        # 필요시 추가 서버 설정
        # "search": {
        #     "command": "python",
        #     "args": ["mcp_servers/search_server.py"],
        #     "transport": "stdio",
        # }
    })

    # MCP 서버로부터 도구 가져오기
    tools = await client.get_tools()
    return tools, client


# ============================================================================
# 4. 그래프 노드 함수들
# ============================================================================

def create_plan(state: AgentState, config) -> dict:
    """
    Step 2: 쿼리 처리 계획을 수립합니다.
    """
    print("\n[단계 2] 계획 수립 중...")

    model = config["configurable"].get("planner_model")
    query = state["query"]

    # 계획 수립을 위한 프롬프트
    planning_prompt = f"""
사용자 쿼리를 분석하고 처리 계획을 수립하세요.

쿼리: {query}

다음 형식으로 계획을 작성하세요:
1. 필요한 단계들을 순서대로 나열
2. 각 단계에서 사용할 도구 식별
3. 최종 응답 생성 방법

계획을 명확하고 구체적으로 작성하세요.
"""

    messages = [HumanMessage(content=planning_prompt)]
    response = model.with_structured_output(Plan).invoke(messages)

    plan_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(response.steps)])
    tools_text = ", ".join(response.required_tools)

    print(f"계획:\n{plan_text}")
    print(f"필요한 도구: {tools_text}\n")

    return {
        "plan": plan_text,
        "messages": [SystemMessage(content=f"실행 계획:\n{plan_text}\n필요한 도구: {tools_text}")]
    }


def execute_agent(state: AgentState, config) -> dict:
    """
    Step 3-4: 계획에 따라 MCP 도구를 호출하고 반복 실행합니다.
    """
    iteration = state.get("iteration", 0) + 1
    print(f"\n[단계 3-4] 에이전트 실행 중... (반복 {iteration}회)")

    model = config["configurable"].get("agent_model")
    messages = state["messages"]

    # 사용자 쿼리가 메시지에 없으면 추가
    if iteration == 1:
        messages = messages + [HumanMessage(content=state["query"])]

    # 모델 호출
    response = model.invoke(messages)

    return {
        "messages": [response],
        "iteration": iteration
    }


def should_continue(state: AgentState) -> Literal["tools", "generate_response", "end"]:
    """
    에이전트가 계속 실행되어야 하는지 결정합니다.
    """
    messages = state["messages"]
    last_message = messages[-1]
    iteration = state.get("iteration", 0)

    # 최대 반복 횟수 제한 (무한 루프 방지)
    MAX_ITERATIONS = 10
    if iteration >= MAX_ITERATIONS:
        print(f"\n[결정] 최대 반복 횟수 도달 ({MAX_ITERATIONS}회)")
        return "generate_response"

    # 도구 호출이 있으면 도구 실행
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"\n[결정] 도구 호출 필요: {len(last_message.tool_calls)}개")
        return "tools"

    # 도구 호출이 없으면 최종 응답 생성
    print("\n[결정] 최종 응답 생성 단계로 이동")
    return "generate_response"


def generate_final_response(state: AgentState, config) -> dict:
    """
    Step 5: 최종 응답을 사용자에게 제공합니다.
    """
    print("\n[단계 5] 최종 응답 생성 중...")

    model = config["configurable"].get("agent_model")
    messages = state["messages"]

    # 최종 응답 생성을 위한 프롬프트 추가
    final_prompt = """
이제 수집한 정보를 바탕으로 사용자의 질문에 대한 최종 답변을 작성하세요.
답변은 명확하고 이해하기 쉽게 작성하세요.
"""

    messages = messages + [SystemMessage(content=final_prompt)]
    response = model.invoke(messages)

    final_answer = response.content if hasattr(response, "content") else str(response)
    print(f"최종 응답: {final_answer}\n")

    return {
        "final_response": final_answer,
        "messages": [response]
    }


# ============================================================================
# 5. 그래프 생성
# ============================================================================

async def create_planning_chatbot_graph():
    """Planning 챗봇 그래프를 생성합니다."""
    # MCP 도구 설정
    tools, client = await setup_mcp_tools()

    # LLM 모델 초기화
    planner_model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    agent_model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    agent_model_with_tools = agent_model.bind_tools(tools)

    # 상태 그래프 초기화
    graph_builder = StateGraph(AgentState)

    # 노드 추가
    graph_builder.add_node("create_plan", create_plan)
    graph_builder.add_node("execute_agent", execute_agent)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_node("generate_final_response", generate_final_response)

    # 엣지 추가
    # START -> 계획 수립
    graph_builder.add_edge(START, "create_plan")

    # 계획 수립 -> 에이전트 실행
    graph_builder.add_edge("create_plan", "execute_agent")

    # 에이전트 실행 -> 조건부 분기
    graph_builder.add_conditional_edges(
        "execute_agent",
        should_continue,
        {
            "tools": "tools",
            "generate_response": "generate_final_response",
            "end": END
        }
    )

    # 도구 실행 -> 에이전트로 다시 돌아감 (반복)
    graph_builder.add_edge("tools", "execute_agent")

    # 최종 응답 -> 종료
    graph_builder.add_edge("generate_final_response", END)

    # 그래프 컴파일
    graph = graph_builder.compile()

    return graph, planner_model, agent_model_with_tools, client


# ============================================================================
# 6. 메인 실행 함수
# ============================================================================

async def run_chatbot(query: str):
    """
    챗봇을 실행합니다.

    Args:
        query: 사용자 질문
    """
    print("=" * 70)
    print("LangGraph Planning MCP 챗봇")
    print("=" * 70)
    print(f"\n[단계 1] 쿼리 입력: {query}\n")

    # 그래프 생성
    graph, planner_model, agent_model, client = await create_planning_chatbot_graph()

    try:
        # 초기 상태 설정
        initial_state = {
            "query": query,
            "plan": "",
            "messages": [],
            "final_response": "",
            "iteration": 0
        }

        # 설정
        config = {
            "configurable": {
                "planner_model": planner_model,
                "agent_model": agent_model,
                "thread_id": "planning_thread"
            }
        }

        # 그래프 실행
        final_state = await graph.ainvoke(initial_state, config)

        print("=" * 70)
        print("실행 완료!")
        print("=" * 70)
        print(f"\n최종 답변:\n{final_state['final_response']}\n")

        return final_state

    finally:
        # MCP 클라이언트 정리
        await client.close()


async def main():
    """메인 함수"""
    # 예시 쿼리들
    queries = [
        "123 곱하기 456을 계산하고, 그 결과에 100을 더한 후 2로 나눈 값은?",
        "(789 + 321) * 3의 결과를 5로 나눈 나머지는?",
        "1부터 10까지의 합을 구한 후, 그 값을 3제곱하면?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n\n### 예시 {i} ###")
        await run_chatbot(query)
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
