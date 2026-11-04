"""포스터용 QR 코드 생성기.

    pip install segno
    python tools/make_qr.py https://chatssc-xxxxx.a.run.app

출력 (기본 폴더: qr_output/)
  qr.svg  : 벡터 형식. 포스터·인쇄용으로 이걸 쓰세요 (크기를 키워도 안 깨짐)
  qr.png  : 미리보기·PPT/한글 문서 삽입용 (고해상도)

오류 보정 수준 'q'(약 25% 훼손까지 복원)를 써서 인쇄 얼룩이나 접힘에도 읽히게 합니다.
"""
import argparse
from pathlib import Path

import segno


def main() -> None:
    parser = argparse.ArgumentParser(description="URL을 QR 코드(SVG/PNG)로 만듭니다.")
    parser.add_argument("url", help="챗봇 주소 (https://로 시작)")
    parser.add_argument("--out-dir", default="qr_output", help="저장 폴더")
    args = parser.parse_args()

    if not args.url.startswith("https://"):
        raise SystemExit("주소가 https:// 로 시작하지 않습니다. Cloud Run 주소를 그대로 넣으세요.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    qr = segno.make_qr(args.url, error="q")  # make_qr: 마이크로 QR 대신 표준 QR 강제
    qr.save(out / "qr.svg", scale=10, border=4)  # border=4: 규격이 요구하는 여백(조용한 영역)
    qr.save(out / "qr.png", scale=20, border=4)

    print(f"저장 완료: {out.resolve()}")
    print(f"QR 버전 {qr.version}, 모듈 수 {qr.symbol_size(border=0)[0]}x{qr.symbol_size(border=0)[1]}")
    print("인쇄 가이드: QR 한 변을 5cm 이상, 주변 여백은 QR 칸 4개 폭 이상 유지하세요.")


if __name__ == "__main__":
    main()
