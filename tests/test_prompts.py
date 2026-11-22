import unittest
from unittest.mock import patch

from test_contracts import S

from voc.prompts import INSTRUCTIONS_DIR, ROLE_FILES, build_instructions


class PromptsTest(unittest.TestCase):
    def test_injects_only_common_and_selected_role(self):
        expected_roles = {
            S.COLLECTOR: "COLLECTOR", S.ANALYST: "ANALYST", S.WRITER_DRAFT: "WRITER",
            S.WRITER_REVISION: "WRITER", S.REVIEWER: "REVIEWER",
            S.REVIEWER_RECHECK: "REVIEWER", S.MANAGER: "MANAGER",
        }
        with patch("socket.socket", side_effect=AssertionError("네트워크 금지")) as socket:
            common = (INSTRUCTIONS_DIR / "VOC_COMMON.md").read_text(encoding="utf-8")
            for stage, role in expected_roles.items():
                with self.subTest(stage=stage):
                    selected = f"VOC_{role}.md"
                    prompt = build_instructions(stage)
                    content = (INSTRUCTIONS_DIR / selected).read_text(encoding="utf-8")
                    self.assertEqual(prompt, f"{common}\n\n{content}")
                    for filename in set(ROLE_FILES.values()) - {selected}:
                        title = (INSTRUCTIONS_DIR / filename).read_text(encoding="utf-8").splitlines()[0]
                        self.assertNotIn(title, prompt)
            socket.assert_not_called()

    def test_human_stages_are_rejected(self):
        for stage in (S.HUMAN_REVIEW, S.FINAL_APPROVAL):
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                build_instructions(stage)
