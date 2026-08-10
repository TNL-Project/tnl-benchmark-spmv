#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

STATS_URL = "https://sparse.tamu.edu/files/ssstats.csv"
MATRIX_URL = "http://sparse-files.engr.tamu.edu/MM"
USER_AGENT = "tnl-suitesparse-downloader/1.0"


@dataclass(frozen=True)
class Matrix:
    group: str
    name: str
    nrows: int
    ncols: int
    nnz: int
    is_real: bool

    @property
    def full_name(self) -> str:
        return f"{self.group}/{self.name}"

    @property
    def url(self) -> str:
        group = urllib.parse.quote(self.group, safe="")
        name = urllib.parse.quote(self.name, safe="")
        return f"{MATRIX_URL}/{group}/{name}.tar.gz"

    def gpu_bytes(self, scalar_bytes: int, index_bytes: int) -> int:
        value_bytes = scalar_bytes if self.is_real else 2 * scalar_bytes
        return (
            self.nnz * (value_bytes + index_bytes)
            + (self.nrows + 1) * index_bytes
            + self.ncols * value_bytes
            + self.nrows * value_bytes
        )


def parse_size(text: str) -> int:
    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)(?:I?B)?\s*",
        text.upper(),
    )
    if not match:
        raise argparse.ArgumentTypeError(
            f"invalid size '{text}' (examples: 512M, 16G, 1.5T)"
        )

    factor = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
    }[match.group(2)]

    return int(float(match.group(1)) * factor)


def format_size(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{x:.3f} {unit}"
        x /= 1024
    raise AssertionError


def render_progress(
    prefix: str,
    current: int,
    total: int | None,
    *,
    width: int = 30,
    final: bool = False,
    format_value: callable = format_size,
) -> None:
    """Show progress towards `total` units of `format_value` (bytes by
    default; pass e.g. str for a plain count): a bar that overwrites itself
    in place on a live terminal, or an occasional plain line when stdout is
    redirected (e.g. to a log file), so a long download doesn't spam it with
    one line per chunk.
    """
    if total:
        fraction = min(1.0, current / total)
        filled = int(width * fraction)
        bar = "#" * filled + "-" * (width - filled)
        line = f"{prefix} [{bar}] {format_value(current)}/{format_value(total)} ({100 * fraction:5.1f}%)"
    else:
        # Server did not report a size (e.g. missing Content-Length): show
        # bytes transferred so far without a percentage.
        line = f"{prefix} {format_value(current)}"

    if sys.stdout.isatty():
        padded = line.ljust(getattr(render_progress, "_last_len", 0))
        sys.stdout.write("\r" + padded + ("\n" if final else ""))
        render_progress._last_len = 0 if final else len(line)
        sys.stdout.flush()
    elif final:
        print(line)


def download(url: str, destination: Path, *, progress_label: str = "  progress:") -> None:
    """Download through destination.part and resume it when possible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    start = partial.stat().st_size if partial.exists() else 0

    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"

    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request) as response:
            append = start > 0 and getattr(response, "status", None) == 206
            content_length = response.headers.get("Content-Length")
            total = start + int(content_length) if content_length else None
            downloaded = start
            last_update = 0.0

            with partial.open("ab" if append else "wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_update >= 0.3:
                        render_progress(progress_label, downloaded, total)
                        last_update = now
            render_progress(progress_label, downloaded, total, final=True)
    except urllib.error.HTTPError as exc:
        if start and exc.code == 416:
            partial.unlink(missing_ok=True)
            return download(url, destination, progress_label=progress_label)
        raise

    partial.replace(destination)


def load_metadata(path: Path) -> tuple[list[Matrix], str]:
    matrices = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        try:
            declared_count = next(reader)[0].strip()
            revision = next(reader)[0].strip()
        except StopIteration as exc:
            raise RuntimeError(f"incomplete metadata file: {path}") from exc

        for line_no, row in enumerate(reader, start=3):
            if len(row) < 6:
                print(f"warning: malformed metadata line {line_no}", file=sys.stderr)
                continue

            try:
                matrices.append(
                    Matrix(
                        group=row[0].strip(),
                        name=row[1].strip(),
                        nrows=int(row[2]),
                        ncols=int(row[3]),
                        nnz=int(row[4]),
                        is_real=bool(int(row[5])),
                    )
                )
            except ValueError:
                print(f"warning: malformed metadata line {line_no}", file=sys.stderr)

    if declared_count.isdigit() and int(declared_count) != len(matrices):
        print(
            f"warning: metadata says {declared_count} matrices, "
            f"but {len(matrices)} were parsed",
            file=sys.stderr,
        )

    return matrices, revision


def extract_matrix(archive: Path, matrix: Matrix, destination: Path) -> None:
    """Extract only <matrix-name>.mtx, not RHS vectors or other archive files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    expected = f"{matrix.name}.mtx"

    with tarfile.open(archive, "r:gz") as tf:
        members = [
            m for m in tf.getmembers()
            if m.isfile() and PurePosixPath(m.name).name == expected
        ]

        if not members:
            raise RuntimeError(f"{expected} not found in {archive}")

        member = min(members, key=lambda m: len(PurePosixPath(m.name).parts))
        source = tf.extractfile(member)
        if source is None:
            raise RuntimeError(f"cannot extract {member.name} from {archive}")

        with source, partial.open("wb") as out:
            shutil.copyfileobj(source, out, length=1024 * 1024)

    partial.replace(destination)


def write_manifest(
    path: Path,
    matrices: list[Matrix],
    scalar_bytes: int,
    index_bytes: int,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ("group", "name", "nrows", "ncols", "nnz",
             "is_real", "gpu_bytes", "gpu_GiB", "url")
        )

        for m in matrices:
            memory = m.gpu_bytes(scalar_bytes, index_bytes)
            writer.writerow(
                (
                    m.group,
                    m.name,
                    m.nrows,
                    m.ncols,
                    m.nnz,
                    int(m.is_real),
                    memory,
                    f"{memory / 1024**3:.6f}",
                    m.url,
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download SuiteSparse matrices whose CSR representation plus "
            "SpMV vectors x and y fit into a given GPU-memory budget."
        )
    )

    parser.add_argument(
        "--max-gpu-memory",
        required=True,
        type=parse_size,
        metavar="SIZE",
        help="maximum estimated GPU memory per matrix, e.g. 4G, 16G, 28G",
    )
    parser.add_argument(
        "--scalar-bytes",
        type=int,
        choices=(4, 8),
        default=8,
        help="real scalar size: 4=float, 8=double (default: 8)",
    )
    parser.add_argument(
        "--index-bytes",
        type=int,
        choices=(4, 8),
        default=4,
        help="signed CSR index size: 4 or 8 bytes (default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./mtx_matrices"),
        help="output directory (default: ./mtx_matrices)",
    )
    parser.add_argument(
        "--include-complex",
        action="store_true",
        help="include complex matrices (their values use 2 * scalar-bytes)",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="keep .tar.gz files after extracting the main .mtx matrix",
    )
    parser.add_argument(
        "--extract-matrices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="extract the .mtx matrix out of each downloaded .tar.gz archive "
        "(default: enabled); use --no-extract-matrices to only download the "
        "archives and leave them packed",
    )
    parser.add_argument(
        "--refresh-metadata",
        action="store_true",
        help="download ssstats.csv even if a cached copy exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only select matrices and write the manifest",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    out = args.output_dir
    matrices_dir = out / "suitesparse"
    metadata = out / "ssstats.csv"
    manifest = out / "selected-matrices.tsv"
    # Without extraction, the archive itself is the final output, so it must
    # not be treated as disposable download scratch space.
    keep_archives = args.keep_archives or not args.extract_matrices
    archive_root = out / (".archives" if keep_archives else ".download")

    out.mkdir(parents=True, exist_ok=True)

    if args.refresh_metadata or not metadata.exists():
        print(f"Downloading metadata: {STATS_URL}")
        download(STATS_URL, metadata)
    else:
        print(f"Using cached metadata: {metadata}")

    matrices, revision = load_metadata(metadata)

    max_index = 2 ** (8 * args.index_bytes - 1) - 1
    selected = []
    skipped_large = skipped_complex = skipped_index = 0

    for matrix in matrices:
        if not args.include_complex and not matrix.is_real:
            skipped_complex += 1
            continue

        if max(matrix.nrows, matrix.ncols, matrix.nnz) > max_index:
            skipped_index += 1
            continue

        if matrix.gpu_bytes(args.scalar_bytes, args.index_bytes) > args.max_gpu_memory:
            skipped_large += 1
            continue

        selected.append(matrix)

    write_manifest(manifest, selected, args.scalar_bytes, args.index_bytes)

    print()
    print(f"SuiteSparse revision:       {revision}")
    print(f"Matrices in metadata:       {len(matrices)}")
    print(f"Selected:                   {len(selected)}")
    print(f"Skipped by memory limit:    {skipped_large}")
    print(f"Skipped complex:            {skipped_complex}")
    print(f"Skipped by index range:     {skipped_index}")
    print(f"Maximum GPU memory:         {format_size(args.max_gpu_memory)}")
    print(f"Scalar / index size:        {args.scalar_bytes} / {args.index_bytes} B")
    print(f"Manifest:                   {manifest}")

    if args.dry_run:
        return 0

    try:
        for i, matrix in enumerate(selected, start=1):
            destination = matrices_dir / matrix.group / f"{matrix.name}.mtx"
            archive = archive_root / matrix.group / f"{matrix.name}.tar.gz"
            # The final output is the extracted .mtx, unless extraction is
            # disabled, in which case the archive itself is what we keep.
            final = destination if args.extract_matrices else archive
            memory = matrix.gpu_bytes(args.scalar_bytes, args.index_bytes)

            print()
            render_progress("Overall progress:", i - 1, len(selected), format_value=str)
            print()
            print(f"[{i:4d}/{len(selected)}] {matrix.full_name}")
            print(f"  shape:          {matrix.nrows} x {matrix.ncols}")
            print(f"  nnz:            {matrix.nnz}")
            print(f"  estimated GPU:  {format_size(memory)}")

            if final.exists() and final.stat().st_size > 0:
                print("  status:         already exists")
                continue

            if not archive.exists() or archive.stat().st_size == 0:
                print("  status:         downloading")
                download(matrix.url, archive, progress_label="  progress:      ")
            else:
                print("  archive:        already exists")

            if args.extract_matrices:
                print("  status:         extracting")
                extract_matrix(archive, matrix, destination)

                if not args.keep_archives:
                    archive.unlink(missing_ok=True)

                print(f"  output:         {destination}")
            else:
                print(f"  output:         {archive}")

        print()
        render_progress(
            "Overall progress:", len(selected), len(selected), final=True, format_value=str
        )

    finally:
        if args.extract_matrices and not args.keep_archives and archive_root.exists():
            # Remove empty directories, but leave partial downloads for resume.
            for directory in sorted(
                (p for p in archive_root.rglob("*") if p.is_dir()),
                key=lambda p: len(p.parts),
                reverse=True,
            ):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            try:
                archive_root.rmdir()
            except OSError:
                pass

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
