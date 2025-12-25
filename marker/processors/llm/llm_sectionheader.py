import json
from dataclasses import dataclass
from typing import List, Tuple, Annotated

from tqdm import tqdm

from marker.logger import get_logger
from marker.processors.llm import BaseLLMComplexBlockProcessor
from marker.processors.llm.llm_utils import (
    inject_analysis_prompt,
    strip_code_fences,
    string_indicates_no_corrections,
)
from marker.schema import BlockTypes
from marker.schema.blocks import Block
from marker.schema.document import Document
from marker.schema.groups import PageGroup
from marker.telemetry import build_marker_trace_headers
from pydantic import BaseModel, Field

logger = get_logger()


@dataclass(frozen=True)
class _NormalizedSectionHeaderResponse:
    correction_needed: bool
    blocks: list


class LLMSectionHeaderProcessor(BaseLLMComplexBlockProcessor):
    analysis_style: Annotated[
        str,
        "How to structure the LLM analysis field: 'summary' or 'auto'.",
    ] = "summary"
    page_prompt = """You're a text correction expert specializing in accurately analyzing complex PDF documents. You will be given a list of all of the section headers from a document, along with their page number and approximate dimensions.  The headers will be formatted like below, and will be presented in order.

```json
[
    {
        "bbox": [x1, y1, x2, y2],
        "width": x2 - x1,
        "height": y2 - y1,
        "page": 0,
        "id": "/page/0/SectionHeader/1",
        "html": "<h1>Introduction</h1>",
    }, ...
]
```

Bboxes have been normalized to 0-1000.

Your goal is to make sure that the section headers have the correct levels (h1, h2, h3, h4, h5, or h6).  If a section header does not have the right level, edit the html to fix it.

Guidelines:
- Edit the blocks to ensure that the section headers have the correct levels.
- Only edit the h1, h2, h3, h4, h5, and h6 tags.  Do not change any other tags or content in the headers.
- Only include the headers that changed in the `blocks` array (if nothing changed, set `correction_needed` to false and `blocks` to []).
- Every header you output needs to have one and only one level tag (h1, h2, h3, h4, h5, or h6).

**Instructions:**
1. Carefully examine the provided section headers and JSON.
2. {{analysis_instruction}}
3. Output a single JSON object (and only JSON) matching this schema:
    {{analysis_schema}}
    - `correction_needed`: boolean
    - `blocks`: array of objects with `id` and `html` (only include changed headers)
4. If `correction_needed` is false, set `blocks` to an empty array.

**Example:**
Input:
Section Headers
```json
[
    {
        "bbox": [x1, y1, x2, y2],
        "id": "/page/0/SectionHeader/1",
        "page": 0,
        "html": "1 Vector Operations",
    },
    {
        "bbox": [x1, y1, x2, y2],
        "id": "/page/0/SectionHeader/2",
        "page": 0,
        "html": "1.1 Vector Addition",
    },
]
```
Output:
```json
{
  "analysis": "The first section header is missing the h1 tag, and the second section header is missing the h2 tag.",
  "correction_needed": true,
  "blocks": [
    {
      "id": "/page/0/SectionHeader/1",
      "html": "<h1>1 Vector Operations</h1>"
    },
    {
      "id": "/page/0/SectionHeader/2",
      "html": "<h2>1.1 Vector Addition</h2>"
    }
  ]
}
```

**Input:**
Section Headers
```json
{{section_header_json}}
```
"""

    def get_selected_blocks(
        self,
        document: Document,
        page: PageGroup,
    ) -> List[dict]:
        selected_blocks = page.structure_blocks(document)
        json_blocks = [
            self.normalize_block_json(block, document, page)
            for block in selected_blocks
        ]
        return json_blocks

    def process_rewriting(
        self, document: Document, section_headers: List[Tuple[Block, dict]]
    ):
        if self.llm_service is None:
            raise ValueError("LLM service is not configured")
        section_header_json = [sh[1] for sh in section_headers]
        for item in section_header_json:
            _, _, page_id, block_type, block_id = item["id"].split("/")
            item["page"] = page_id
            item["width"] = item["bbox"][2] - item["bbox"][0]
            item["height"] = item["bbox"][3] - item["bbox"][1]
            del item["block_type"]  # Not needed, since they're all section headers

        prompt_template = inject_analysis_prompt(self.page_prompt, self.analysis_style)
        prompt = prompt_template.replace(
            "{{section_header_json}}", json.dumps(section_header_json)
        )
        headers = build_marker_trace_headers(
            source_path=document.filepath,
            processor=self.__class__.__name__,
            block_id=str(document.pages[0].id),
            page_id=document.pages[0].page_id,
            extra={"SectionHeaderCount": len(section_header_json)},
        )
        response = self.llm_service(
            prompt, None, document.pages[0], SectionHeaderSchema, extra_headers=headers
        )
        logger.debug(f"Got section header reponse from LLM: {response}")

        normalized = self._normalize_response(response)
        if normalized is None:
            logger.warning("LLM did not return a valid response")
            return

        if not normalized.correction_needed:
            return

        if not normalized.blocks:
            return

        self.handle_rewrites(normalized.blocks, document)

    def _normalize_response(self, response) -> _NormalizedSectionHeaderResponse | None:
        if response is None:
            return None

        if isinstance(response, str):
            # TODO: is stripping code fences necessary?
            text = strip_code_fences(response)
            if string_indicates_no_corrections(text):
                return _NormalizedSectionHeaderResponse(
                    correction_needed=False, blocks=[]
                )
            try:
                response = json.loads(text)
            except Exception:
                return None

        if isinstance(response, list):
            return _NormalizedSectionHeaderResponse(
                correction_needed=len(response) > 0, blocks=response
            )

        if not isinstance(response, dict):
            return None

        blocks = response.get("blocks", [])
        if isinstance(blocks, str):
            try:
                blocks = json.loads(strip_code_fences(blocks))
            except Exception:
                blocks = []

        correction_needed = response.get("correction_needed", None)
        if correction_needed is None:
            correction_needed = bool(blocks)

        return _NormalizedSectionHeaderResponse(
            correction_needed=bool(correction_needed),
            blocks=blocks if isinstance(blocks, list) else [],
        )

    def rewrite_blocks(self, document: Document):
        # Don't show progress if there are no blocks to process
        section_headers = [
            (block, self.normalize_block_json(block, document, page))
            for page in document.pages
            for block in page.structure_blocks(document)
            if block.block_type == BlockTypes.SectionHeader
        ]
        if len(section_headers) == 0:
            return

        pbar = tqdm(
            total=1,
            desc=f"Running {self.__class__.__name__}",
            disable=self.disable_tqdm,
        )

        self.process_rewriting(document, section_headers)
        pbar.update(1)
        pbar.close()


class BlockSchema(BaseModel):
    id: str
    html: str


class SectionHeaderSchema(BaseModel):
    analysis: str = ""
    correction_needed: bool = False
    blocks: List[BlockSchema] = Field(default_factory=list)
