"""PDF → 청크 → 임베딩 → Qdrant 적재.

사용법 (프로젝트 루트에서):
    python -m app.ingest            # data/documents/*.pdf 적재 (같은 파일명은 교체)
    python -m app.ingest --reset    # 컬렉션을 비우고 전부 다시 적재 (스키마 변경 시)

문서 유형별 처리:
  - 조문형 본문   : '제N조(제목)' 단위로 분할 (없으면 길이 기준 분할)
  - 신·구조문 대비표 : 행마다 '개정 전/후'를 분리. 시행일 기준 현행본만 status=current,
                     나머지는 status=inactive로 저장해 검색에서 제외
  - 입학정원 표   : 연도 제목을 붙여 '학과별 한 줄 문장'으로 변환
  - 그 외 표      : 헤더:값 형식의 행 문장으로 변환
"""
import argparse
import hashlib
import logging
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pymupdf

from app import config, embedding
from app import qdrant as store

logger = logging.getLogger("app.ingest")

# 본문 조문 분할: 줄 시작 + '제N조(제목)'. 제목 괄호가 없는 "제5조에 따라" 같은 인용은 제외
ARTICLE_START = re.compile(r"^제\s*\d+\s*조(?:\s*의\s*\d+)?\s*\([^)]*\)", re.MULTILINE)
# 표 셀 선두의 조항명. PDF에서 '제'가 누락되는 경우("52조의 3")도 허용
CELL_ARTICLE = re.compile(r"제?\s*\d+\s*조(?:\s*의\s*\d+)?\s*(?:\([^)]*\))?")
MIN_CHUNK_CHARS = 30
PANEL_COLS = 5  # 입학정원 표: [수업연한, 학과/계열, 주간, 야간, 계] 패널이 가로로 반복


# ───────────────────────── 공통 유틸 ─────────────────────────
def clean_cell(cell) -> str:
    return re.sub(r"\s+", " ", str(cell or "")).strip()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def category_of_file(path: Path) -> str:
    for keyword, category in config.CATEGORY_BY_FILENAME.items():
        if keyword in path.stem:
            return category
    return config.DEFAULT_CATEGORY


def split_with_offsets(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP):
    """긴 텍스트를 (시작 위치, 조각) 목록으로 분할합니다."""
    text = text.strip()
    if len(text) <= size:
        return [(0, text)]
    pieces, start = [], 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            pieces.append((start, piece))
        if start + size >= len(text):
            break
        start += size - overlap
    return pieces


def split_long_text(text: str, size: int = config.CHUNK_SIZE) -> list[str]:
    return [piece for _, piece in split_with_offsets(text, size)]


# ───────────────────────── PDF 추출 ─────────────────────────
def extract_body_text(page, table_bboxes: list[tuple]) -> str:
    """표 영역 밖의 단어만 모아 줄 단위 텍스트로 복원합니다."""
    words = []
    for w in page.get_text("words"):
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        if not any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in table_bboxes):
            words.append(w)

    lines, current, line_y = [], [], None
    for w in sorted(words, key=lambda w: (w[1], w[0])):
        if line_y is not None and abs(w[1] - line_y) > 3:
            lines.append(current)
            current = []
        if not current:
            line_y = w[1]
        current.append(w)
    if current:
        lines.append(current)
    return "\n".join(" ".join(w[4] for w in sorted(line, key=lambda w: w[0])) for line in lines).strip()


def find_table_title(page, bbox) -> str:
    """표 바로 위 한 줄(예: '2027학년도 입학정원')을 표 제목으로 사용합니다."""
    above = [w for w in page.get_text("words") if w[3] <= bbox[1] + 1 and bbox[1] - w[3] < 80]
    if not above:
        return ""
    bottom = max(w[3] for w in above)
    line = sorted((w for w in above if abs(w[3] - bottom) <= 3), key=lambda w: w[0])
    return " ".join(w[4] for w in line)


def extract_tables(page) -> list[dict]:
    try:
        found = page.find_tables().tables
    except Exception:
        logger.warning("표 탐지 실패 (page %s) — 본문으로만 처리합니다", page.number + 1)
        return []

    tables = []
    for table in found:
        try:
            raw_rows = table.extract()
        except Exception:
            continue
        rows = [[clean_cell(c) for c in row] for row in raw_rows or [] if row]
        rows = [r for r in rows if any(r)]
        if rows:
            tables.append({"rows": rows, "bbox": tuple(table.bbox), "title": find_table_title(page, table.bbox)})
    return tables


def extract_pdf(path: Path) -> list[dict]:
    pages = []
    with pymupdf.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            tables = extract_tables(page)
            body = extract_body_text(page, [t["bbox"] for t in tables])
            pages.append({"page": number, "body": body, "tables": tables})
    return pages


# ───────────────────────── 조문형 본문 ─────────────────────────
def build_full_text(pages: list[dict]):
    parts, ranges, pos = [], [], 0
    for p in pages:
        text = p["body"].strip()
        if not text:
            continue
        ranges.append({"page": p["page"], "start": pos, "end": pos + len(text)})
        parts.append(text + "\n")
        pos += len(text) + 1
    return "".join(parts), ranges


def get_page_range(start: int, end: int, ranges: list[dict]):
    covered = [r["page"] for r in ranges if start < r["end"] and end > r["start"]]
    if not covered:
        return None, None, ""
    first, last = min(covered), max(covered)
    return first, last, str(first) if first == last else f"{first}-{last}"


def split_by_article(full_text: str) -> list[dict]:
    starts = [(m.start(), m.group(0).strip()) for m in ARTICLE_START.finditer(full_text)]
    if not starts or starts[0][0] > 0:
        starts.insert(0, (0, None))  # 첫 조문 앞의 서문/전문
    sections = []
    for i, (start, article) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(full_text)
        text = full_text[start:end].strip()
        if text:
            sections.append({"text": text, "start": start, "article": article})
    return sections


def article_records(pages: list[dict]) -> list[dict]:
    full_text, ranges = build_full_text(pages)
    records = []
    for section in split_by_article(full_text):
        for offset, text in split_with_offsets(section["text"]):
            if len(text) < MIN_CHUNK_CHARS:  # '【별표1】' 같은 제목 조각 제외
                continue
            start = section["start"] + offset
            page_start, page_end, page = get_page_range(start, start + len(text), ranges)
            records.append(
                {
                    "text": text,
                    "article": section["article"],
                    "content_type": "article",
                    "page": page,
                    "page_start": page_start,
                    "page_end": page_end,
                }
            )
    return records


# ───────────────────────── 신·구조문 대비표 ─────────────────────────
def is_revision_table(rows: list[list[str]]) -> bool:
    header = rows[0]
    return len(header) == 2 and bool(re.search(r"변경\s*전", header[0]) and re.search(r"변경\s*후", header[1]))


def parse_year_month(header: str) -> Optional[date]:
    m = re.search(r"\((\d{4})\.\s*(\d{1,2})", header)
    return date(int(m[1]), int(m[2]), 1) if m else None


def revision_records(rows: list[list[str]], today: date) -> list[dict]:
    """대비표의 각 셀을 개별 청크로 만들고, 시행일 기준 현행 여부를 status로 기록합니다."""
    before_header, after_header = rows[0]
    effective = parse_year_month(after_header)
    after_in_force = effective is None or effective <= today
    labels = (re.sub(r"변경\s*전", "개정 전", before_header), re.sub(r"변경\s*후", "개정 후", after_header))

    records = []
    for before, after in rows[1:]:
        for side, cell in enumerate((before, after)):  # side 1 == 개정 후
            if not cell:
                continue
            in_force = (side == 1) == after_in_force
            match = CELL_ARTICLE.match(cell)
            text = re.sub(r"\s*([①-⑩])", r"\n\1", cell).strip()  # 항 번호 앞에서 줄바꿈
            for piece in split_long_text(text):
                records.append(
                    {
                        "text": f"[{labels[side]}] {piece}",
                        "article": match.group(0).strip() if match else None,
                        "status": "current" if in_force else "inactive",
                        "version_label": labels[side],
                    }
                )
    return records


# ───────────────────────── 일반 표 / 입학정원 표 ─────────────────────────
def is_enrollment_table(rows: list[list[str]], title: str) -> bool:
    return "입학정원" in title and len(rows[0]) % PANEL_COLS == 0


def _count(value: str) -> str:
    return "없음" if value in ("", "-") else f"{value}명"


def enrollment_lines(rows: list[list[str]]) -> list[str]:
    """[수업연한, 학과, 주간, 야간, 계] 패널이 가로로 반복되는 표를 학과별 한 줄로 변환합니다."""
    width = len(rows[0])
    panels = width // PANEL_COLS
    level = [""] * panels  # 병합 셀(수업연한)은 빈 칸으로 추출되므로 위 값을 이어받음
    lines = []
    for row in rows:
        row = row + [""] * (width - len(row))
        for p in range(panels):
            lv, name, day, night, total = row[p * PANEL_COLS : (p + 1) * PANEL_COLS]
            stats = f"주간 {_count(day)}, 야간 {_count(night)}, 계 {_count(total)}"
            if "합" in lv and "계" in lv:
                lines.append(f"합계: {stats}")
                continue
            if not name or name == "학과/계열":  # 빈 칸 또는 헤더
                continue
            if lv:
                level[p] = lv
            lines.append(f"{level[p]} {name}: {stats}".strip())
    return lines


def generic_table_lines(rows: list[list[str]]) -> list[str]:
    header = rows[0]
    return [
        "; ".join(f"{header[i] or f'열{i + 1}'}: {v}" for i, v in enumerate(row[: len(header)]) if v)
        for row in rows[1:]
        if any(row)
    ]


def group_lines(lines: list[str], header: str, size: int = config.CHUNK_SIZE) -> list[str]:
    """줄 단위로 묶되 청크마다 표 제목 헤더를 붙입니다(행이 중간에 끊기지 않음)."""
    chunks, current = [], header
    for line in lines:
        if current != header and len(current) + len(line) + 1 > size:
            chunks.append(current)
            current = header
        current += "\n" + line
    if current != header:
        chunks.append(current)
    return chunks


def table_records(rows: list[list[str]], title: str) -> list[dict]:
    lines = enrollment_lines(rows) if is_enrollment_table(rows, title) else generic_table_lines(rows)
    header = f"[{title}]" if title else "[표]"
    return [{"text": text, "table_title": title} for text in group_lines(lines, header)]


def table_and_revision_records(pages: list[dict], today: date) -> list[dict]:
    records = []
    for p in pages:
        for table in p["tables"]:
            rows = table["rows"]
            if is_revision_table(rows):
                items, content_type = revision_records(rows, today), "revision"
            else:
                items, content_type = table_records(rows, table["title"]), "table"
            for item in items:
                records.append(
                    {**item, "content_type": content_type, "page": str(p["page"]), "page_start": p["page"], "page_end": p["page"]}
                )
    return records


# ───────────────────────── 조립 ─────────────────────────
def process_pdf(path: Path, today: Optional[date] = None) -> tuple[list[dict], list[str]]:
    """PDF 1개 → (payload 청크 목록, 임베딩용 텍스트 목록)."""
    today = today or date.today()
    doc_id, category = file_hash(path), category_of_file(path)
    pages = extract_pdf(path)
    records = article_records(pages) + table_and_revision_records(pages, today)

    chunks, embed_texts = [], []
    for i, record in enumerate(records):
        chunk = {
            "status": "current",
            "document_type": "규정",
            "category": category,
            "document_id": doc_id,
            "document_name": path.name,
            "chunk_id": f"{doc_id}_{i}",
            "chunk_index": i,
            **record,
        }
        chunks.append(chunk)
        # 문서명·조항/표 제목을 임베딩 텍스트에 덧붙여 검색 정확도를 높임 (저장되는 text는 원문 그대로)
        context = " ".join(filter(None, [path.stem, chunk.get("article") or chunk.get("table_title")]))
        embed_texts.append(f"{context}\n{chunk['text']}")
    return chunks, embed_texts


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="PDF 문서를 Qdrant에 적재합니다.")
    parser.add_argument("--reset", action="store_true", help="컬렉션을 삭제 후 다시 만들어 전부 재적재")
    parser.add_argument("--dir", type=Path, default=config.DOCUMENT_DIR, help="PDF 폴더 (기본: data/documents)")
    args = parser.parse_args()

    pdfs = sorted(args.dir.glob("*.pdf"))
    if not pdfs:
        logger.error("적재할 PDF가 없습니다: %s", args.dir)
        return 1

    if args.reset:
        store.reset_collection()
    else:
        store.init_collection()

    total = 0
    for pdf in pdfs:
        chunks, embed_texts = process_pdf(pdf)
        if not chunks:
            logger.warning("추출된 내용이 없습니다(스캔본이면 OCR 필요): %s", pdf.name)
            continue
        vectors = embedding.encode_batch(embed_texts)
        store.delete_document(pdf.name)
        total += store.upsert_documents(chunks, vectors)
        logger.info("%s → %d개 청크 (category=%s)", pdf.name, len(chunks), chunks[0]["category"])
    logger.info("✨ 적재 완료: 총 %d개 청크", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
