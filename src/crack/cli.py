"""명령줄 진입점."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from . import config as config_mod
from . import export, pipeline, stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="crack",
        description="드론 정사영상에서 색상 표시된 균열을 추출해 CAD 데이터로 변환합니다.",
    )
    p.add_argument("image", help="입력 정사영상 (GeoTIFF)")
    p.add_argument("-c", "--config", default=None, help="설정 TOML 경로")
    p.add_argument("-o", "--out-dir", default="out", help="출력 폴더 (기본: out)")
    p.add_argument("--prefix", default=None, help="출력 파일명 접두사 (기본: 입력 파일명)")
    p.add_argument("--tile-size", type=int, default=4096, help="타일 한 변 픽셀 수")
    p.add_argument("--no-dxf", action="store_true", help="DXF 출력 생략")
    p.add_argument("--no-tsv", action="store_true", help="TSV 출력 생략")
    p.add_argument("--no-preview", action="store_true", help="미리보기 PNG 생략")
    p.add_argument(
        "--preview-by-color",
        action="store_true",
        help="색상(등급)별로 미리보기 PNG 를 한 장씩 따로 저장",
    )
    p.add_argument("--no-stats", action="store_true", help="속성값 분석 TSV 생략")
    p.add_argument(
        "--sort",
        choices=config_mod.SORT_KEYS,
        default=None,
        help="출력 정렬 기준 (기본: 설정의 analysis.sort_by)",
    )
    p.add_argument("--gsd", type=float, default=None,
                   help="mm/픽셀. 좌표계 없는 영상(JPG 등)에 사용")
    p.add_argument("-q", "--quiet", action="store_true", help="진행 로그 숨김")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def progress(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    try:
        cfg = config_mod.load(args.config)
        if args.gsd is not None:
            cfg = replace(cfg, fallback_gsd_mm=args.gsd)
    except (FileNotFoundError, ValueError) as e:
        print(f"[설정 오류] {e}", file=sys.stderr)
        return 2

    try:
        result = pipeline.run(args.image, cfg, tile_size=args.tile_size, progress=progress)
    except FileNotFoundError as e:
        print(f"[입력 오류] {e}", file=sys.stderr)
        return 2

    # 정렬한 뒤 내보내야 DXF·TSV 의 순서가 분석 결과와 일치한다
    segments = stats.sort_segments(result.segments, cfg, args.sort)
    summary = stats.analyze(segments, cfg)

    out_dir = Path(args.out_dir)
    prefix = args.prefix or Path(args.image).stem
    written: list[Path] = []

    if not args.no_dxf:
        written.append(export.write_dxf(segments, cfg, result.raster,
                                       out_dir / f"{prefix}.dxf"))
    if not args.no_tsv:
        written.append(export.write_tsv(segments, cfg, out_dir / f"{prefix}.tsv"))
    if not args.no_stats:
        written.append(export.write_stats_tsv(summary, cfg,
                                              out_dir / f"{prefix}_stats.tsv"))
    if not args.no_preview:
        written.append(export.write_preview(segments, cfg, result.raster,
                                            out_dir / f"{prefix}_preview.png"))
    if args.preview_by_color:
        written.extend(export.write_previews_by_color(
            segments, cfg, result.raster, out_dir, prefix))

    if not args.quiet:
        print()
        print(stats.format_report(summary, cfg))
        print()
        drops = ", ".join(f"{k} {v}" for k, v in result.dropped.items() if v)
        if drops:
            print(f"  노이즈로 제외: {drops}")
        print(f"  정렬 기준: {args.sort or cfg.analysis.sort_by}")
        print(f"  처리 시간: {result.elapsed_s:.1f}s")
        print()
        for w in written:
            print(f"  -> {w}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
