import sys
import unittest
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from voc.loader import AIR_ITEM, consultation_chunks, read_chunks


def synthetic_chunks(data, codes, *, code_only=False):
    header = data.splitlines()[0] + "\n"

    def read_csv(_path, **kwargs):
        assert set(kwargs["usecols"]) <= {"ACCIDENT_NO", "RCPT_YMD", "ITEM_CODE", "품목(ITEM_NAME)"}
        return original_read_csv(StringIO(data), **kwargs)

    original_read_csv = pd.read_csv
    with (
        patch.object(Path, "open", return_value=BytesIO(header.encode("utf-8"))),
        patch("voc.loader.pd.read_csv", side_effect=read_csv),
    ):
        return list(consultation_chunks(
            "synthetic.csv", codes, code_only=code_only, chunksize=2, encoding="utf-8",
        ))


class LoaderTest(unittest.TestCase):
    def test_exact_name_and_verified_code_classification(self):
        codes = {}
        names_header = "ACCIDENT_NO,RCPT_YMD,ITEM_CODE,품목(ITEM_NAME),ACCIDENT_TITLE\n"
        named = synthetic_chunks(names_header + (
            "01,20200101,A,항공 여객운송서비스,합성 제목\n"
            f"02,20200102,B,{AIR_ITEM} 부가,합성 제목\n"
            "03,20200103,C,,합성 제목\n"
            f"04,20200104,D,{AIR_ITEM},합성 제목\n"
        ), codes)
        self.assertEqual(list(named[0].columns), ["ACCIDENT_NO", "RCPT_YMD", "is_air"])
        self.assertEqual(named[0]["ACCIDENT_NO"].tolist(), ["01", "02"])
        self.assertEqual(named[0]["is_air"].tolist(), [True, False])
        self.assertTrue(pd.isna(named[1]["is_air"].iloc[0]))
        synthetic_chunks(names_header + "05,20200105,D,다른품목,합성 제목\n", codes)

        hazard = synthetic_chunks(
            "ACCIDENT_NO,RCPT_YMD,ITEM_CODE,ANS_CONTENT\n"
            "11,20200201,A,합성 답변\n12,20200202,B,합성 답변\n"
            "13,20200203,D,합성 답변\n14,20200204,Z,합성 답변\n15,20200205,,합성 답변\n",
            codes, code_only=True,
        )
        self.assertEqual(hazard[0]["is_air"].tolist(), [True, False])
        self.assertEqual(sum(int(chunk["is_air"].isna().sum()) for chunk in hazard), 3)

    def test_blocked_and_unlisted_columns_rejected_before_csv_read(self):
        with patch("voc.loader.pd.read_csv") as reader:
            for column in ("ACCIDENT_CONTENT", "답변내용(ITEM_NAME)", "기타"):
                with self.subTest(column=column), self.assertRaises(ValueError):
                    read_chunks("synthetic.csv", ["ACCIDENT_NO", column])
            reader.assert_not_called()
