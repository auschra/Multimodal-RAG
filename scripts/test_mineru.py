# src/scripts/benchmark_mineru.py
import logging
import time
from pathlib import Path
from statistics import mean, median, stdev
import concurrent.futures
from src.clients.mineru_client import MinerUClient
import concurrent.futures


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

def benchmark_single(pdf_path: Path, client: MinerUClient) -> dict:
    """Parse one PDF and return timing info."""
    result = {
        "file": pdf_path.name,
        "size_mb": pdf_path.stat().st_size / (1024 * 1024),
        "parse_seconds": None,
        "read_seconds": None,
        "num_blocks": None,
        "num_pages": None,
        "error": None,
    }

    try:
        t0 = time.perf_counter()
        parse_result = client.parse(pdf_path)
        t1 = time.perf_counter()

        content = client.read_content_list(parse_result.output_dir, pdf_path.stem)
        t2 = time.perf_counter()

        result["parse_seconds"] = t1 - t0
        result["read_seconds"] = t2 - t1
        result["num_blocks"] = len(content)
        result["num_pages"] = max((b.get("page_idx", 0) for b in content), default=0) + 1
    except Exception as e:
        logger.exception(f"Failed on {pdf_path.name}")
        result["error"] = str(e)

    return result


def main():
    
    client = MinerUClient()
    raw_doc_dir = Path("/storage/bulk/raw_docs")
    pdfs = sorted(file for file in raw_doc_dir.iterdir())

    MAX_CONCURRENCY = 2 
    logger.info(f"Benchmarking {len(pdfs)} pdfs with {MAX_CONCURRENCY} workers")

    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        future_to_pdf = {executor.submit(benchmark_single, pdf, client): pdf for pdf in pdfs}
        
        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf = future_to_pdf[future]
            try:
                result = future.result()
                results.append(result)
                if result["error"]:
                    logger.warning(f"  -> FAILED: {pdf.name} - {result['error'][:100]}")
                else:
                    logger.info(
                        f"  -> SUCCESS: {pdf.name} | parse={result['parse_seconds']:.1f}s  "
                        f"pages={result['num_pages']}  blocks={result['num_blocks']}"
                    )
            except Exception as exc:
                logger.error(f"{pdf.name} generated an exception: {exc}")

    # (Keep your existing summary code below this)

    # Summary
    successful = [r for r in results if r["error"] is None]
    if not successful:
        logger.error("All documents failed. Check MinerU container logs.")
        return

    parse_times = [r["parse_seconds"] for r in successful]
    pages = [r["num_pages"] for r in successful]

    print("\n" + "=" * 60)
    print(f"Summary: {len(successful)}/{len(results)} successful")
    print("=" * 60)
    print(f"Parse time  —  mean: {mean(parse_times):.1f}s  "
          f"median: {median(parse_times):.1f}s  "
          f"min: {min(parse_times):.1f}s  "
          f"max: {max(parse_times):.1f}s")
    if len(parse_times) > 1:
        print(f"Parse time stdev: {stdev(parse_times):.1f}s")
    print(f"Pages per doc — mean: {mean(pages):.0f}  median: {median(pages):.0f}")
    total_pages = sum(pages)
    total_time = sum(parse_times)
    print(f"Throughput: {total_pages / total_time:.1f} pages/sec  "
          f"({total_time / len(successful):.1f}s per doc average)")

    # Project to 500 docs
    projected = (mean(parse_times)) * 500
    print(f"\nProjected time for 500 docs sequentially: "
          f"{projected / 60:.0f} minutes ({projected / 3600:.1f} hours)")


if __name__ == "__main__":
    main()