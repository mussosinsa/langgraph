"""Utility to convert PDFs to SKT A.Dot X 4.0 fine-tuning format using opendataloader-pdf.

This script wraps the official ``opendataloader-pdf`` package to extract text from
all PDF files inside a directory.  The extracted text is chunked and exported as a
JSONL file that follows the conversational structure required by SKT A.Dot X 4.0
fine-tuning datasets.

Example::

    python examples/opendataloader_pdf_to_adot.py ./docs ./output.jsonl \
        --recursive --chunk-size 1200 --overlap 200

The script assumes that ``opendataloader-pdf`` is installed in the environment.
Refer to the package documentation for additional CLI flags if you need more
control over the PDF parsing stage.
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from opendataloader_pdf import run as run_opendataloader

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract text from Korean PDFs with opendataloader-pdf and convert the "
            "result into the SKT A.Dot X 4.0 fine-tuning dataset format."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory that contains the source PDF files.",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Destination JSONL file that will contain the generated conversations.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for PDFs recursively instead of only reading the top-level directory.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1400,
        help="Number of characters per assistant response.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Number of characters that overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "당신은 SKT A.Dot X 4.0을 사용하는 한국어 조수입니다. 문서의 내용을 정확하고 "
            "자연스러운 한국어로 전달하세요."
        ),
        help="System 프롬프트 문구.",
    )
    parser.add_argument(
        "--user-template",
        default="{source_name} 문서의 원문을 보여줘.",
        help=(
            "User 메시지 템플릿. ``str.format`` 규칙을 사용하며 ``source_name``(필수), "
            "``chunk_index`` 및 ``total_chunks`` 변수를 사용할 수 있습니다."
        ),
    )
    parser.add_argument(
        "--assistant-template",
        default="{chunk}",
        help=(
            "Assistant 메시지 템플릿. ``chunk``(필수), ``source_name``, ``chunk_index`` 및 "
            "``total_chunks`` 변수를 사용할 수 있습니다."
        ),
    )
    parser.add_argument(
        "--keep-line-breaks",
        action="store_true",
        help="opendataloader-pdf 실행 시 줄바꿈을 유지합니다.",
    )
    parser.add_argument(
        "--content-safety-off",
        nargs="*",
        default=None,
        help="opendataloader-pdf 내용 안전 필터를 비활성화할 때 사용할 옵션 리스트.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="표준 출력에 기록할 로그 레벨.",
    )
    parser.add_argument(
        "--cli-debug",
        action="store_true",
        help="opendataloader-pdf Java CLI 로그를 표시합니다.",
    )
    return parser.parse_args()


def find_pdf_files(root: Path, recursive: bool) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"입력 디렉터리를 찾을 수 없습니다: {root}")
    if recursive:
        files = sorted(p for p in root.rglob("*.pdf") if p.is_file())
    else:
        files = sorted(p for p in root.glob("*.pdf") if p.is_file())
    LOGGER.debug("Found %d PDF files", len(files))
    return files


def read_pdf_with_opendataloader(
    pdf_path: Path,
    keep_line_breaks: bool,
    content_safety_off: Optional[List[str]],
    cli_debug: bool,
) -> List[bytes]:
    with tempfile.TemporaryDirectory(prefix="opendataloader_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        run_kwargs: Dict[str, object] = {
            "output_folder": tmp_dir,
            "keep_line_breaks": keep_line_breaks,
            "no_json": False,
            "debug": cli_debug,
        }
        if content_safety_off:
            run_kwargs["content_safety_off"] = ",".join(content_safety_off)
        LOGGER.debug("Running opendataloader-pdf on %s", pdf_path)
        run_opendataloader(str(pdf_path), **run_kwargs)
        json_files = sorted(tmp_path.glob("*.json"))
        if not json_files:
            raise RuntimeError(
                f"opendataloader-pdf에서 JSON 결과를 찾을 수 없습니다: {pdf_path}"
            )
        return [path.read_bytes() for path in json_files]


def iter_text_nodes(node: object) -> Iterator[str]:
    if isinstance(node, dict):
        content = node.get("content")
        if isinstance(content, str):
            yield content
        text_value = node.get("text")
        if isinstance(text_value, str):
            yield text_value
        kids = node.get("kids")
        if isinstance(kids, list):
            for child in kids:
                yield from iter_text_nodes(child)
        # 탐색 가능한 다른 중첩 필드를 순회합니다.
        for key, value in node.items():
            if key in {"content", "text", "kids"}:
                continue
            if isinstance(value, (dict, list)):
                yield from iter_text_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_text_nodes(item)


def extract_text_from_json_bytes(json_bytes: bytes) -> str:
    data = json.loads(json_bytes.decode("utf-8"))
    fragments = [frag.strip() for frag in iter_text_nodes(data) if frag and frag.strip()]
    return "\n\n".join(fragments)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size는 0보다 커야 합니다.")
    if overlap >= chunk_size:
        raise ValueError("overlap은 chunk_size보다 작아야 합니다.")
    cleaned = "\n".join(line.strip() for line in text.splitlines()).strip()
    if not cleaned:
        return []
    chunks: List[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(length, start + chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(0, end - overlap)
    return chunks


def format_template(template: str, **values: object) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:  # pragma: no cover - template misuse guard
        missing = exc.args[0]
        raise ValueError(f"템플릿에 필요한 변수 '{missing}' 가 제공되지 않았습니다.") from exc


def build_conversations(
    pdf_path: Path,
    chunks: List[str],
    system_prompt: str,
    user_template: str,
    assistant_template: str,
) -> Iterable[Dict[str, object]]:
    total_chunks = len(chunks)
    for index, chunk in enumerate(chunks):
        metadata = {
            "source": str(pdf_path),
            "chunk_index": index,
            "total_chunks": total_chunks,
        }
        values = {
            "chunk": chunk,
            "source_name": pdf_path.stem,
            "chunk_index": index + 1,
            "total_chunks": total_chunks,
        }
        conversation = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": format_template(user_template, **values),
                },
                {
                    "role": "assistant",
                    "content": format_template(assistant_template, **values),
                },
            ],
            "metadata": metadata,
        }
        yield conversation


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    pdf_files = find_pdf_files(args.input_dir, args.recursive)
    if not pdf_files:
        raise FileNotFoundError("지정한 경로에서 PDF 파일을 찾지 못했습니다.")
    conversations: List[Dict[str, object]] = []
    for pdf_file in pdf_files:
        LOGGER.info("Processing %s", pdf_file)
        json_payloads = read_pdf_with_opendataloader(
            pdf_file,
            keep_line_breaks=args.keep_line_breaks,
            content_safety_off=args.content_safety_off,
            cli_debug=args.cli_debug,
        )
        combined_texts = [extract_text_from_json_bytes(payload) for payload in json_payloads]
        merged_text = "\n\n".join(filter(None, combined_texts)).strip()
        if not merged_text:
            LOGGER.warning("%s 에서 텍스트를 추출하지 못했습니다.", pdf_file)
            continue
        chunks = chunk_text(merged_text, args.chunk_size, args.overlap)
        if not chunks:
            LOGGER.warning("%s 에서 유효한 청크를 생성하지 못했습니다.", pdf_file)
            continue
        conversations.extend(
            build_conversations(
                pdf_file,
                chunks,
                system_prompt=args.system_prompt,
                user_template=args.user_template,
                assistant_template=args.assistant_template,
            )
        )
    if not conversations:
        raise RuntimeError("어떠한 대화 데이터도 생성되지 않았습니다.")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as file:
        for conversation in conversations:
            file.write(json.dumps(conversation, ensure_ascii=False))
            file.write("\n")
    LOGGER.info(
        "변환 완료: %d 개의 대화 항목을 %s 파일에 저장했습니다.",
        len(conversations),
        args.output_file,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
