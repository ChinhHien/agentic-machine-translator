"""Pipeline package: orchestration, assembly, checkpoints, consistency.

``polyglot_lc.pipeline`` used to be a single ``pipeline.py`` module. It is now a
package so each concern has one file: the coordinator owns the workflow, the
assembler owns output, the checkpoint store owns durability and the consistency
checker owns cross-chunk terminology. The public names are unchanged, so
``from polyglot_lc.pipeline import TranslationPipeline`` still works.
"""

from polyglot_lc.pipeline.assembler import (
    assemble,
    assemble_partial,
    block_translations,
    output_path,
    output_paths,
    write_atomic,
)
from polyglot_lc.pipeline.checkpoint import CheckpointStore, ChunkCheckpoint
from polyglot_lc.pipeline.consistency import (
    ConsistencyChecker,
    ConsistencyReport,
    Inconsistency,
    find_inconsistencies,
)
from polyglot_lc.pipeline.coordinator import PipelineResult, TranslationPipeline
from polyglot_lc.pipeline.pdf_output import PdfOutputError, RebuildStats, rebuild_pdf

__all__ = [
    "CheckpointStore",
    "ChunkCheckpoint",
    "ConsistencyChecker",
    "ConsistencyReport",
    "Inconsistency",
    "PdfOutputError",
    "PipelineResult",
    "RebuildStats",
    "TranslationPipeline",
    "assemble",
    "assemble_partial",
    "block_translations",
    "find_inconsistencies",
    "output_path",
    "output_paths",
    "rebuild_pdf",
    "write_atomic",
]
