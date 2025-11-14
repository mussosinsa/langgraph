"""
LangGraph와 MCP Tool을 사용한 기본 챗봇 에이전트

이 예제는 ReAct 패턴을 사용하여 MCP 도구와 통합된 챗봇을 구현합니다.
"""
import asyncio
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# 1. State 정의
class State(TypedDict):
    """에이전트의 상태를 정의합니다."""
    messages: Annotated[list[BaseMessage], add_messages]


# 2. MCP 클라이언트 및 도구 설정
async def setup_mcp_tools():
    """MCP 서버에서 도구를 가져옵니다."""
    client = MultiServerMCPClient({
        # 예시: 계산기 MCP 서버
        "calculator": {
            "command": "python",
            "args": ["mcp_servers/calculator_server.py"],
            "transport": "stdio",
        },
        # 예시: 검색 MCP 서버 (필요시 추가)
        # "search": {
        #     "command": "python",
        #     "args": ["mcp_servers/search_server.py"],
        #     "transport": "stdio",
        # }
    })

    # MCP 서버로부터 도구 가져오기
    tools = await client.get_tools()
    return tools, client


# 3. 모델 호출 노드 정의
def call_model(state: State, config):
    """LLM을 호출하여 응답을 생성합니다."""
    model = config["configurable"].get("model")
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


# 4. 그래프 생성
async def create_chatbot_graph():
    """챗봇 그래프를 생성합니다."""
    # MCP 도구 설정
    tools, client = await setup_mcp_tools()

    # LLM 모델 초기화
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    model_with_tools = model.bind_tools(tools)

    # 상태 그래프 초기화
    graph_builder = StateGraph(State)

    # 노드 추가
    graph_builder.add_node("call_model", call_model)
    graph_builder.add_node("tools", ToolNode(tools))

    # 엣지 추가
    graph_builder.add_edge(START, "call_model")
    graph_builder.add_conditional_edges(
        "call_model",
        tools_condition,  # 도구 호출이 필요한지 확인
    )
    graph_builder.add_edge("tools", "call_model")

    # 그래프 컴파일
    graph = graph_builder.compile()

    return graph, model_with_tools, client


# 5. 메인 실행 함수
async def main():
    """메인 실행 함수"""
    print("=== LangGraph MCP 챗봇 시작 ===\n")

    # 그래프 생성
    graph, model, client = await create_chatbot_graph()

    try:
        # 예시 쿼리
        queries = [
            "123 곱하기 456은 얼마인가요?",
            "(789 + 321) / 2를 계산해주세요.",
        ]

        for query in queries:
            print(f"질문: {query}")
            print("-" * 50)

            # 그래프 실행
            state = {
                "messages": [HumanMessage(content=query)]
            }

            config = {
                "configurable": {
                    "model": model,
                    "thread_id": "example_thread"
                }
            }

            # 스트리밍으로 결과 출력
            async for event in graph.astream(state, config, stream_mode="values"):
                if "messages" in event and event["messages"]:
                    last_message = event["messages"][-1]
                    if hasattr(last_message, "content") and last_message.content:
                        print(f"응답: {last_message.content}")

            print("\n" + "=" * 50 + "\n")

    finally:
        # MCP 클라이언트 정리
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
