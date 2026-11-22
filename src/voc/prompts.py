from pathlib import Path

from .contracts import StageName

INSTRUCTIONS_DIR = Path(__file__).resolve().parents[2] / "instructions"
ROLE_FILES = {
    StageName.COLLECTOR: "VOC_COLLECTOR.md",
    StageName.ANALYST: "VOC_ANALYST.md",
    StageName.WRITER_DRAFT: "VOC_WRITER.md",
    StageName.WRITER_REVISION: "VOC_WRITER.md",
    StageName.REVIEWER: "VOC_REVIEWER.md",
    StageName.REVIEWER_RECHECK: "VOC_REVIEWER.md",
    StageName.MANAGER: "VOC_MANAGER.md",
}


def build_instructions(stage: StageName, instructions_dir: Path = INSTRUCTIONS_DIR) -> str:
    role_file = ROLE_FILES.get(stage)
    if role_file is None:
        raise ValueError("모델 단계에만 역할 지침을 적용할 수 있습니다.")
    directory = Path(instructions_dir)
    common = (directory / "VOC_COMMON.md").read_text(encoding="utf-8")
    role = (directory / role_file).read_text(encoding="utf-8")
    return f"{common}\n\n{role}"
