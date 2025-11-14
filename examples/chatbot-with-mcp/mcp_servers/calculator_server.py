"""
계산기 MCP 서버

이 서버는 기본적인 수학 연산 도구를 제공합니다.
"""
import asyncio
import sys
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# 서버 인스턴스 생성
app = Server("calculator")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """사용 가능한 도구 목록을 반환합니다."""
    return [
        Tool(
            name="add",
            description="두 숫자를 더합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "첫 번째 숫자"
                    },
                    "b": {
                        "type": "number",
                        "description": "두 번째 숫자"
                    }
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="subtract",
            description="첫 번째 숫자에서 두 번째 숫자를 뺍니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "첫 번째 숫자"
                    },
                    "b": {
                        "type": "number",
                        "description": "두 번째 숫자"
                    }
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="multiply",
            description="두 숫자를 곱합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "첫 번째 숫자"
                    },
                    "b": {
                        "type": "number",
                        "description": "두 번째 숫자"
                    }
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="divide",
            description="첫 번째 숫자를 두 번째 숫자로 나눕니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "첫 번째 숫자 (분자)"
                    },
                    "b": {
                        "type": "number",
                        "description": "두 번째 숫자 (분모)"
                    }
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="power",
            description="첫 번째 숫자를 두 번째 숫자만큼 거듭제곱합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "base": {
                        "type": "number",
                        "description": "밑"
                    },
                    "exponent": {
                        "type": "number",
                        "description": "지수"
                    }
                },
                "required": ["base", "exponent"]
            }
        ),
        Tool(
            name="modulo",
            description="첫 번째 숫자를 두 번째 숫자로 나눈 나머지를 구합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "첫 번째 숫자"
                    },
                    "b": {
                        "type": "number",
                        "description": "두 번째 숫자"
                    }
                },
                "required": ["a", "b"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """도구를 실행합니다."""
    try:
        if name == "add":
            result = arguments["a"] + arguments["b"]
            return [TextContent(
                type="text",
                text=f"{arguments['a']} + {arguments['b']} = {result}"
            )]

        elif name == "subtract":
            result = arguments["a"] - arguments["b"]
            return [TextContent(
                type="text",
                text=f"{arguments['a']} - {arguments['b']} = {result}"
            )]

        elif name == "multiply":
            result = arguments["a"] * arguments["b"]
            return [TextContent(
                type="text",
                text=f"{arguments['a']} × {arguments['b']} = {result}"
            )]

        elif name == "divide":
            if arguments["b"] == 0:
                return [TextContent(
                    type="text",
                    text="오류: 0으로 나눌 수 없습니다"
                )]
            result = arguments["a"] / arguments["b"]
            return [TextContent(
                type="text",
                text=f"{arguments['a']} ÷ {arguments['b']} = {result}"
            )]

        elif name == "power":
            result = arguments["base"] ** arguments["exponent"]
            return [TextContent(
                type="text",
                text=f"{arguments['base']} ^ {arguments['exponent']} = {result}"
            )]

        elif name == "modulo":
            if arguments["b"] == 0:
                return [TextContent(
                    type="text",
                    text="오류: 0으로 나눈 나머지를 구할 수 없습니다"
                )]
            result = arguments["a"] % arguments["b"]
            return [TextContent(
                type="text",
                text=f"{arguments['a']} mod {arguments['b']} = {result}"
            )]

        else:
            return [TextContent(
                type="text",
                text=f"오류: 알 수 없는 도구 '{name}'"
            )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"오류: {str(e)}"
        )]


async def main():
    """MCP 서버를 시작합니다."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
