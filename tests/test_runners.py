import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_contracts import S, V, sanitized

from voc.contracts import CheckedText, StageInput
from voc.runners import FakeRunner, SdkRunner


class RunnersTest(unittest.TestCase):
    def setUp(self):
        self.socket = self.enterContext(patch("socket.socket", side_effect=AssertionError("네트워크 금지")))
        self.addCleanup(self.socket.assert_not_called)

    def test_fake_writer_uses_input_stats_and_keeps_draft_in_revision(self):
        runner = FakeRunner()
        data = sanitized()
        draft_input = StageInput(S.WRITER_DRAFT, 0, 1, data)
        draft = runner.run_stage(draft_input)
        self.assertEqual(draft, runner.run_stage(draft_input))
        draft_text = draft.output_text
        assert draft_text is not None
        self.assertTrue(draft_text.startswith("DRY_RUN 합성 출력 — 실제 모델 결과 아님\n"))
        for title in ("개요", "VOC 현황", "핵심 원인", "개선방안", "실행 우선순위", "위험·한계", "결론"):
            self.assertIn(f"## {title}\n", draft_text)
        self.assertIn("건수 12", draft_text)
        self.assertIn("기간 2020-01", draft_text)
        self.assertNotIn("비율", draft_text)
        self.assertIn("가설(DRY_RUN 예시)", draft_text)
        self.assertEqual((draft.artifact_version, draft.calls_used, draft.input_chars),
                         (1, 1, draft_input.total_chars()))

        prior = ((S.WRITER_DRAFT, CheckedText(draft_text, True)),)
        revision = runner.run_stage(StageInput(S.WRITER_REVISION, 1, 2, data, prior))
        revision_text = revision.output_text
        assert revision_text is not None
        self.assertTrue(revision_text.startswith(draft_text + "\n\n## 변경점\n"))
        self.assertEqual((revision.round_no, revision.artifact_version), (1, 2))

    def test_fake_review_verdicts_and_default(self):
        runner = FakeRunner((V.NEEDS_REVISION, V.UNDETERMINED))
        first = runner.run_stage(StageInput(S.REVIEWER, 0, 1, sanitized()))
        self.assertIs(first.verdict, V.NEEDS_REVISION)
        assert first.output_text is not None
        self.assertIn("‘위험·한계’ 절에", first.output_text)
        recheck = StageInput(S.REVIEWER_RECHECK, 1, 2, sanitized())
        second = runner.run_stage(recheck)
        self.assertEqual((second.verdict, second.artifact_version), (V.UNDETERMINED, 2))
        self.assertIs(runner.run_stage(recheck).verdict, V.APPROVABLE)

    def test_sdk_requires_approval(self):
        with self.assertRaises(RuntimeError):
            SdkRunner("test-model").run_stage(StageInput(S.COLLECTOR, 0, 0, sanitized()))

    def test_sdk_default_model_is_configured_luna(self):
        self.assertEqual(SdkRunner().model, "gpt-6-luna")

    def test_sdk_refuses_without_output_token_limit(self):
        step = StageInput(S.COLLECTOR, 0, 0, sanitized())
        for bad in (None, 0, True):
            with self.assertRaises(RuntimeError):
                SdkRunner("test-model", approved_live=True, max_output_tokens=bad).run_stage(step)  # pyright: ignore[reportArgumentType]

    def _stub_sdk(self):
        sdk = SimpleNamespace(Agent=Mock(), RunConfig=Mock(), Runner=Mock(), ModelSettings=Mock(),
                              OpenAIResponsesModel=Mock())
        usage = SimpleNamespace(requests=1, input_tokens=100, output_tokens=20)
        sdk.Runner.run_sync.side_effect = lambda agent, payload, **options: SimpleNamespace(
            final_output=sdk.Agent.call_args.kwargs["output_type"](V.APPROVABLE, "검수 완료"),
            context_wrapper=SimpleNamespace(usage=usage),
        )
        return sdk

    def test_sdk_uses_injected_client_model(self):
        sdk, client = self._stub_sdk(), object()
        with patch.dict(sys.modules, {"agents": sdk}):
            SdkRunner("test-model", approved_live=True, max_output_tokens=500, client=client).run_stage(
                StageInput(S.REVIEWER, 0, 1, sanitized()),
            )
        sdk.OpenAIResponsesModel.assert_called_once_with(model="test-model", openai_client=client)
        self.assertIs(sdk.Agent.call_args.kwargs["model"], sdk.OpenAIResponsesModel.return_value)

    def test_sdk_run_configuration_and_structured_review_with_stub(self):
        sdk = self._stub_sdk()
        runner = SdkRunner("test-model", approved_live=True, max_output_tokens=500)
        with patch.dict(sys.modules, {"agents": sdk}):
            result = runner.run_stage(StageInput(S.REVIEWER, 0, 1, sanitized()))
        self.assertEqual(runner.usage_log,
                         [{"stage": "reviewer", "requests": 1, "input_tokens": 100, "output_tokens": 20}])
        sdk.RunConfig.assert_called_once_with(tracing_disabled=True, trace_include_sensitive_data=False)
        sdk.ModelSettings.assert_called_once_with(max_tokens=500)
        self.assertEqual(sdk.Agent.call_args.kwargs["model"], "test-model")
        self.assertIs(sdk.Agent.call_args.kwargs["model_settings"], sdk.ModelSettings.return_value)
        call = sdk.Runner.run_sync.call_args
        self.assertEqual(call.kwargs, {"max_turns": 1, "run_config": sdk.RunConfig.return_value})
        self.assertIn('"건수": 12', call.args[1])
        self.assertIn("# Reviewer", sdk.Agent.call_args.kwargs["instructions"])
        self.assertEqual((result.verdict, result.output_text, result.output_tokens),
                         (V.APPROVABLE, "검수 완료", 20))
