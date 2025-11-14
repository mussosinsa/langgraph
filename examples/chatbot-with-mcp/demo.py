"""
간단한 데모 스크립트

Planning MCP 챗봇의 간단한 사용 예제입니다.
"""
import asyncio
from planning_mcp_chatbot import run_chatbot


async def main():
    """메인 함수"""
    print("=" * 70)
    print("LangGraph MCP 챗봇 데모")
    print("=" * 70)
    print()
    print("이 데모는 LangGraph와 MCP Tool을 사용하여")
    print("복잡한 수학 문제를 자동으로 해결하는 과정을 보여줍니다.")
    print()
    print("=" * 70)
    print()

    # 대화형 모드
    print("대화형 모드를 시작합니다. 'quit' 또는 'exit'를 입력하면 종료됩니다.")
    print()

    while True:
        try:
            # 사용자 입력 받기
            query = input("\n질문을 입력하세요: ").strip()

            # 종료 명령어 체크
            if query.lower() in ['quit', 'exit', '종료', 'q']:
                print("\n챗봇을 종료합니다. 감사합니다!")
                break

            # 빈 입력 체크
            if not query:
                print("질문을 입력해주세요.")
                continue

            # 챗봇 실행
            await run_chatbot(query)

        except KeyboardInterrupt:
            print("\n\n챗봇을 종료합니다. 감사합니다!")
            break
        except Exception as e:
            print(f"\n오류가 발생했습니다: {e}")
            print("다시 시도해주세요.")


if __name__ == "__main__":
    # 환경 변수 체크
    import os

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("=" * 70)
        print("경고: ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.")
        print()
        print("다음 명령어로 설정하세요:")
        print('  export ANTHROPIC_API_KEY="your-api-key-here"')
        print("=" * 70)
        print()

    asyncio.run(main())
