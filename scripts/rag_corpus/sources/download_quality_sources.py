from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import argparse
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from scripts.rag_corpus.common.io import project_root, write_json
from scripts.rag_corpus.common.progress import StageProgress
from scripts.rag_corpus.sources.source_registry import QUALITY_CORPUS_BY_ID


@dataclass(frozen=True)
class WebPageSource:
    url: str
    relative_path: str
    min_bytes: int = 500


@dataclass(frozen=True)
class SourceDownloadSpec:
    source_id: str
    git_url: str | None = None
    pages: tuple[WebPageSource, ...] = field(default_factory=tuple)


DOWNLOAD_SPECS: dict[str, SourceDownloadSpec] = {
    "wilke_fundamentals": SourceDownloadSpec(
        source_id="wilke_fundamentals",
        pages=(
            WebPageSource("https://clauswilke.com/dataviz/", "index.html", 2000),
            WebPageSource("https://clauswilke.com/dataviz/visualizing-amounts.html", "visualizing_amounts.html", 2000),
            WebPageSource("https://clauswilke.com/dataviz/histograms-density-plots.html", "histograms_density_plots.html", 2000),
            WebPageSource("https://clauswilke.com/dataviz/overlapping-points.html", "overlapping_points.html", 2000),
            WebPageSource("https://clauswilke.com/dataviz/color-pitfalls.html", "color_pitfalls.html", 2000),
            WebPageSource("https://clauswilke.com/dataviz/proportional-ink.html", "proportional_ink.html", 2000),
            WebPageSource("https://clauswilke.com/dataviz/figure-titles-captions.html", "figure_titles_captions.html", 2000),
            WebPageSource("https://clauswilke.com/dataviz/balance-data-context.html", "balance_data_context.html", 2000),
        ),
    ),
    "from_data_to_viz": SourceDownloadSpec(
        source_id="from_data_to_viz",
        git_url="https://github.com/holtzy/data_to_viz.git",
        pages=(
            WebPageSource("https://www.data-to-viz.com/", "site_pages/index.html", 2000),
            WebPageSource("https://www.data-to-viz.com/caveats.html", "site_pages/caveats.html", 2000),
            WebPageSource("https://www.data-to-viz.com/graph/barplot.html", "site_pages/barplot.html", 2000),
            WebPageSource("https://www.data-to-viz.com/graph/line.html", "site_pages/line.html", 2000),
            WebPageSource("https://www.data-to-viz.com/graph/scatter.html", "site_pages/scatter.html", 2000),
            WebPageSource("https://www.data-to-viz.com/graph/histogram.html", "site_pages/histogram.html", 2000),
            WebPageSource("https://www.data-to-viz.com/graph/boxplot.html", "site_pages/boxplot.html", 2000),
            WebPageSource("https://www.data-to-viz.com/graph/violin.html", "site_pages/violin.html", 2000),
            WebPageSource("https://www.data-to-viz.com/graph/treemap.html", "site_pages/treemap.html", 2000),
            WebPageSource("https://www.data-to-viz.com/graph/heatmap.html", "site_pages/heatmap.html", 2000),
        ),
    ),
    "ft_visual_vocabulary": SourceDownloadSpec(
        source_id="ft_visual_vocabulary",
        git_url="https://github.com/Financial-Times/chart-doctor.git",
        pages=(
            WebPageSource(
                "https://raw.githubusercontent.com/Financial-Times/chart-doctor/main/visual-vocabulary/README.md",
                "visual-vocabulary/README.md",
                1000,
            ),
        ),
    ),
    "uk_analysis_colours": SourceDownloadSpec(
        source_id="uk_analysis_colours",
        pages=(
            WebPageSource(
                "https://analysisfunction.civilservice.gov.uk/policy-store/data-visualisation-colours-in-charts/",
                "index.html",
                2000,
            ),
        ),
    ),
    "uk_charts_checklist": SourceDownloadSpec(
        source_id="uk_charts_checklist",
        pages=(
            WebPageSource(
                "https://analysisfunction.civilservice.gov.uk/policy-store/charts-a-checklist/",
                "index.html",
                2000,
            ),
        ),
    ),
    "urban_institute_style_guide": SourceDownloadSpec(
        source_id="urban_institute_style_guide",
        pages=(
            WebPageSource("https://urbaninstitute.github.io/graphics-styleguide/", "index.html", 2000),
        ),
    ),
    "chartability": SourceDownloadSpec(
        source_id="chartability",
        git_url="https://github.com/chartability/POUR-CAF.git",
        pages=(
            WebPageSource("https://chartability.github.io/POUR-CAF/", "site_pages/index.html", 2000),
        ),
    ),
}


class SourceDownloadError(RuntimeError):
    pass


def _raw_dir(root: Path, source_id: str) -> Path:
    source = QUALITY_CORPUS_BY_ID[source_id]
    return root / source.raw_dir


def _run_git(args: list[str], *, cwd: Path | None = None, timeout_seconds: float) -> str:
    try:
        completed = subprocess.run(  # noqa: S603
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SourceDownloadError(f"Git command failed before completion: git {' '.join(args)}: {exc}") from exc
    output = completed.stdout.strip()
    if completed.returncode != 0:
        raise SourceDownloadError(f"Git command failed: git {' '.join(args)}\n{output}")
    return output


def _ensure_git_clone(url: str, destination: Path, *, refresh: bool, timeout_seconds: float) -> dict[str, object]:
    if destination.exists() and refresh:
        shutil.rmtree(destination)
    if destination.exists() and (destination / ".git").exists():
        output = _run_git(["pull", "--ff-only"], cwd=destination, timeout_seconds=timeout_seconds)
        return {"status": "updated", "path": str(destination), "message": output}
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = _run_git(["clone", "--depth", "1", url, str(destination)], timeout_seconds=timeout_seconds)
    return {"status": "downloaded", "path": str(destination), "message": output}


def _download_page(page: WebPageSource, output_path: Path, *, refresh: bool, timeout_seconds: float) -> dict[str, object]:
    if output_path.exists() and not refresh and output_path.stat().st_size >= page.min_bytes:
        return {"status": "exists", "url": page.url, "path": str(output_path), "bytes": output_path.stat().st_size}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        page.url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            payload = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SourceDownloadError(f"Cannot download {page.url} -> {output_path}: {exc}") from exc
    if len(payload) < page.min_bytes:
        raise SourceDownloadError(
            f"Downloaded source is too small: {page.url} -> {output_path}: "
            f"{len(payload)} bytes, expected at least {page.min_bytes}"
        )
    output_path.write_bytes(payload)
    return {"status": "downloaded", "url": page.url, "path": str(output_path), "bytes": len(payload)}


def download_quality_sources(
    root: Path,
    *,
    sources: list[str] | None = None,
    refresh: bool = False,
    timeout_seconds: float = 60.0,
) -> dict[str, object]:
    selected = list(sources or DOWNLOAD_SPECS.keys())
    unknown = sorted(source for source in selected if source not in DOWNLOAD_SPECS)
    if unknown:
        raise ValueError(f"No download spec for sources: {', '.join(unknown)}")

    report: dict[str, object] = {"sources": {}, "selected_sources": selected}
    progress = StageProgress("download-sources", total=len(selected))
    for source_id in selected:
        spec = DOWNLOAD_SPECS[source_id]
        source_report: dict[str, object] = {"status": "ok", "items": []}
        target_dir = _raw_dir(root, source_id)
        if spec.git_url:
            item = _ensure_git_clone(spec.git_url, target_dir, refresh=refresh, timeout_seconds=timeout_seconds)
            source_report["items"].append({"kind": "git", "url": spec.git_url, **item})  # type: ignore[index]
        for page in spec.pages:
            item = _download_page(page, target_dir / page.relative_path, refresh=refresh, timeout_seconds=timeout_seconds)
            source_report["items"].append({"kind": "web_page", **item})  # type: ignore[index]
        report["sources"][source_id] = source_report  # type: ignore[index]
        progress.update(extra=f"{source_id}: ok")
    progress.finish()
    write_json(root / "rag_corpus/reports/source_download_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/cache practical ViRAGE visrag corpus sources.")
    parser.add_argument("--sources", nargs="*", choices=sorted(DOWNLOAD_SPECS.keys()), default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()
    report = download_quality_sources(
        project_root(),
        sources=args.sources,
        refresh=args.refresh,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
