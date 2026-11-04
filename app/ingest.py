"""PDF → 청크 → 임베딩 → Qdrant 적재.

사용법 (프로젝트 루트에서):
    python -m app.ingest --reset --limit 5        # 기존 데이터를 모두 지우고 data/documents 의 앞 5개만 시험 적재
    python -m app.ingest --skip-existing          # 전체 적재 (중단돼도 같은 명령으로 이어서)
    python -m app.ingest --dir data/new_docs      # 다른 폴더(하위 폴더 포함)만 추가 적재 — 기존 문서는 유지
    python -m app.ingest --group 4                # 폴더의 모든 PDF를 대분류 4로 저장 (번호가 없는 파일용)

파일명 규칙:  '4-1-8 교원인사 규정.pdf'
  → 규정번호 4-1-8 / 대분류 4(인사·행정) / 규정명 '교원인사 규정'
  이름이 '폐지'로 끝나면(예: '..._폐지.pdf') 폐지된 규정이라 기본적으로 적재하지 않습니다.

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
import time
from collections import Counter
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


REGULATION_NO = re.compile(r"^(\d+)-(\d+)-(\d+)[\s._\-]*(.*)$")
ABOLISHED = re.compile(r"[\s_\-]*폐지\s*$")


def parse_filename(path: Path, forced_group: Optional[str] = None) -> dict:
    """'4-1-8 교원인사 규정.pdf' → 규정번호·대분류·규정명. 이름 끝이 '폐지'이면 폐지된 규정으로 표시합니다."""
    stem = path.stem.strip()
    abolished = bool(ABOLISHED.search(stem))
    clean = ABOLISHED.sub("", stem).strip()
    match = REGULATION_NO.match(clean)
    if match:
        first, second, serial, name = match.groups()
        number, subgroup, group_no = f"{first}-{second}-{serial}", f"{first}-{second}", first
        name = name.strip() or clean
    else:
        number = subgroup = group_no = None
        name = clean
    if forced_group:
        group_no = forced_group
    if group_no not in config.GROUPS:
        group_no = None
    return {
        "regulation_no": number,
        "regulation_name": name,
        "group_no": group_no,
        "subgroup_no": subgroup,
        "category": config.GROUPS.get(group_no, config.DEFAULT_CATEGORY),
        "abolished": abolished,
    }


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
def process_pdf(
    path: Path, today: Optional[date] = None, group: Optional[str] = None
) -> tuple[list[dict], list[str]]:
    """PDF 1개 → (payload 청크 목록, 임베딩용 텍스트 목록)."""
    today = today or date.today()
    doc_id = file_hash(path)
    meta = parse_filename(path, forced_group=group)
    pages = extract_pdf(path)
    records = article_records(pages) + table_and_revision_records(pages, today)

    chunks, embed_texts = [], []
    for i, record in enumerate(records):
        chunk = {
            "status": "current",
            "document_type": "규정",
            "category": meta["category"],
            "group_no": meta["group_no"],
            "subgroup_no": meta["subgroup_no"],
            "regulation_no": meta["regulation_no"],
            "regulation_name": meta["regulation_name"],
            "document_id": doc_id,
            "document_name": path.name,
            "chunk_id": f"{doc_id}_{i}",
            "chunk_index": i,
            **record,
        }
        if meta["abolished"]:
            chunk["status"] = "inactive"  # 폐지된 규정은 검색 대상에서 제외
        chunks.append(chunk)
        # 분야·규정명·조항/표 제목을 임베딩 텍스트 앞에 붙여 검색 정확도를 높임 (저장되는 text는 원문 그대로)
        context = " ".join(
            filter(None, [meta["category"], meta["regulation_name"], chunk.get("article") or chunk.get("table_title")])
        )
        embed_texts.append(f"{context}\n{chunk['text']}")
    return chunks, embed_texts


def find_pdfs(directory: Path) -> list[Path]:
    """하위 폴더까지 포함해 PDF를 찾습니다 (확장자 대소문자 무시)."""
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def find_duplicate_names(pdfs: list[Path]) -> list[str]:
    counts = Counter(p.name for p in pdfs)
    return sorted(name for name, n in counts.items() if n > 1)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="PDF 문서를 Qdrant에 적재합니다.")
    parser.add_argument("--dir", type=Path, default=config.DOCUMENT_DIR, help="PDF 폴더 (하위 폴더 포함, 기본: data/documents)")
    parser.add_argument("--group", choices=sorted(config.GROUPS), help="폴더의 모든 PDF를 이 대분류(1~9)로 저장 (생략하면 파일명 번호로 판단)")
    parser.add_argument("--include-abolished", action="store_true", help="이름이 '폐지'로 끝나는 규정도 비현행(inactive)으로 저장 (검색에는 나오지 않음)")
    parser.add_argument("--skip-existing", action="store_true", help="이미 같은 내용으로 적재된 파일은 건너뜀 (중단 후 이어서 실행)")
    parser.add_argument("--limit", type=int, help="앞에서부터 N개만 처리 (시험용)")
    parser.add_argument("--reset", action="store_true", help="컬렉션을 삭제 후 전부 재적재 (기본 폴더에서만 사용 가능)")
    args = parser.parse_args()

    if args.reset and args.dir.resolve() != Path(config.DOCUMENT_DIR).resolve():
        logger.error("--reset은 기본 폴더(%s)에서만 쓸 수 있습니다. 다른 폴더에 쓰면 기존 문서가 모두 사라집니다.", config.DOCUMENT_DIR)
        return 1

    pdfs = find_pdfs(args.dir)
    if not pdfs:
        logger.error("적재할 PDF가 없습니다: %s", args.dir)
        return 1
    duplicates = find_duplicate_names(pdfs)
    if duplicates:
        logger.error("파일 이름이 같은 PDF가 있습니다(서로 덮어쓰게 되므로 이름을 바꿔 주세요): %s", ", ".join(duplicates))
        return 1
    if args.limit:
        pdfs = pdfs[: args.limit]

    if args.reset:
        store.reset_collection()
    else:
        store.init_collection()

    done, skipped, empty, failed, abolished = [], [], [], [], []
    by_category = Counter()
    total_chunks = 0
    started = time.time()
    for i, pdf in enumerate(pdfs, 1):
        label = f"[{i}/{len(pdfs)}] {pdf.name}"
        if parse_filename(pdf, forced_group=args.group)["abolished"] and not args.include_abolished:
            abolished.append(pdf.name)
            logger.info("%s → 폐지된 규정이라 제외", label)
            continue
        uploading = False
        try:
            if args.skip_existing and store.document_exists(pdf.name, file_hash(pdf)):
                skipped.append(pdf.name)
                logger.info("%s → 이미 적재됨, 건너뜀", label)
                continue
            chunks, embed_texts = process_pdf(pdf, group=args.group)
            if not chunks:
                empty.append(pdf.name)
                logger.warning("%s → 추출된 내용이 없습니다 (스캔본이면 OCR 필요)", label)
                continue
            vectors = embedding.encode_batch(embed_texts)
            uploading = True
            store.delete_document(pdf.name)
            total_chunks += store.upsert_documents(chunks, vectors)
            done.append(pdf.name)
            by_category[chunks[0]["category"]] += 1
            logger.info("%s → %d개 청크 (분야=%s)", label, len(chunks), chunks[0]["category"])
        except Exception:
            logger.exception("%s → 실패", label)
            failed.append(pdf.name)
            if uploading:  # 일부만 올라간 문서가 '적재됨'으로 오인되지 않도록 정리
                try:
                    store.delete_document(pdf.name)
                except Exception:
                    logger.warning("실패한 문서 정리에 실패: %s", pdf.name)

    logger.info("===== 적재 결과 (%.1f분) =====", (time.time() - started) / 60)
    logger.info(
        "성공 %d개 (청크 %d개) / 건너뜀 %d개 / 폐지 제외 %d개 / 내용 없음 %d개 / 실패 %d개",
        len(done), total_chunks, len(skipped), len(abolished), len(empty), len(failed),
    )
    if by_category:
        logger.info("분야별 문서 수: %s", ", ".join(f"{k} {v}개" for k, v in sorted(by_category.items())))
    for title, names in (("내용 없음 (OCR 필요 가능)", empty), ("실패", failed)):
        if names:
            logger.warning("%s: %s", title, ", ".join(names))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
