import requests
from pathlib import Path
from src.config import load_config
import json
from dataclasses import dataclass
import zipfile

cfg = load_config()

@dataclass
class ParseResult:
    output_dir: Path
    markdown_path: Path
    content_list_path: Path
    middle_json_path: Path

# currently send pdf files to mineruclient -> switch to using file paths when possible

class MinerUClient:
    def __init__(self,
        api_url = "http://localhost:8000",
        host_input_dir: Path = Path("/storage/bulk/raw_docs"),
        container_input_dir = Path("/input"),
        host_output_dir: Path = Path("/datasets/scratch/mineru-output"),
        container_output_dir = Path("/output"),
    ):
        self.api_url = api_url.rstrip("/")
        self.host_input_dir = host_input_dir
        self.container_input_dir = container_input_dir
        self.host_output_dir = host_output_dir
        self.container_output_dir = container_output_dir                                       

    # Translates dirs from storage drive to docker container
    def _get_container_path(self, host_filepath):
        relative_path = host_filepath.relative_to(self.host_input_dir)

        return str(self.container_input_dir / relative_path)
        

    def parse(self, host_filepath):
        if not host_filepath.exists():
            raise FileNotFoundError(f"No file at {host_filepath}")

        doc_stem = host_filepath.stem
        host_output = self.host_output_dir / doc_stem
        host_output.mkdir(parents=True, exist_ok=True)

        # Upload the PDF as multipart, request the artifacts we care about back
        with open(host_filepath, "rb") as f:
            try:
                response = requests.post(
                    f"{self.api_url}/file_parse",
                    files={"files": (host_filepath.name, f, "application/pdf")},
                    data={
                        "backend": "pipeline",
                        "parse_method": "auto",
                        "lang_list": "en",
                        "formula_enable": "true",
                        "table_enable": "true",
                        "return_md": "true",
                        "return_middle_json": "true",
                        "return_content_list": "true",
                        "return_images": "true",
                        "response_format_zip": "true",  
                    },
                    timeout=(10, 1800),
                )
                response.raise_for_status()

            except requests.HTTPError as e:
                detail = response.text[:1000] if response.content else "(no body)"
                raise RuntimeError(
                    f"MinerU failed on {host_filepath.name} "
                    f"(HTTP {response.status_code}): {detail}"
                ) from e

        # Save the returned ZIP and extract
        zip_path = host_output / f"{doc_stem}.zip"
        zip_path.write_bytes(response.content)

        with zipfile.ZipFile(zip_path) as z:
            z.extractall(host_output)
        zip_path.unlink()  # remove zip, keep extracted files

        # Dynamically find the extracted artifacts
        try:
            md_path = list(host_output.rglob("*.md"))[0]
            content_path = list(host_output.rglob("*_content_list.json"))[0]
            middle_path = list(host_output.rglob("*_middle.json"))[0]
        except IndexError:
            raise FileNotFoundError(f"MinerU ZIP extracted, but expected artifacts are missing in {host_output}")

        return ParseResult(
            output_dir=host_output,
            markdown_path=md_path,
            content_list_path=content_path,
            middle_json_path=middle_path,
        )
    
    # Read output file firom dir
    def read_content_list(self, output_host_dir: Path, pdf_stem: str):
        # Recursively search for the JSON file to bypass ZIP nesting inconsistencies
        matches = list(output_host_dir.rglob(f"{pdf_stem}_content_list.json"))
        
        if not matches:
            # Fallback in case MinerU drops the stem prefix in future versions
            matches = list(output_host_dir.rglob("content_list.json"))
            
        if not matches:
            raise FileNotFoundError(f"Could not find content list JSON anywhere in {output_host_dir}")
            
        return json.loads(matches[0].read_text())