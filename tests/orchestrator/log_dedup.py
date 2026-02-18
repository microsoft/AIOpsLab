# test_log_deduplication.py
import os
import unittest
from aiopslab.orchestrator.actions import log_deduplication as ld


class TestTimestampDetection(unittest.TestCase):
    def test_find_timestamp_spans_iso(self):
        line = "2025-09-24 18:41:09 some log message"
        spans = ld.find_timestamp_spans(line, ld.DEFAULT_TS_RX)
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        self.assertEqual(line[start:end], "2025-09-24 18:41:09")

    def test_find_timestamp_spans_multiple(self):
        line = "Start 2025-09-24T18:41:09Z end 2025-09-24T19:00:00Z"
        spans = ld.find_timestamp_spans(line, ld.DEFAULT_TS_RX)
        self.assertEqual(len(spans), 2)

    def test_find_timestamp_spans_no_match(self):
        line = "no timestamp here"
        spans = ld.find_timestamp_spans(line, ld.DEFAULT_TS_RX)
        self.assertEqual(spans, [])


class TestMakeBlocks(unittest.TestCase):
    def test_make_blocks_basic(self):
        lines = ["a", "b", "c", "d", "e"]
        blocks = ld.make_blocks(lines, 2)
        self.assertEqual(blocks, ["a\nb", "c\nd", "e"])

    def test_make_blocks_invalid(self):
        with self.assertRaises(ValueError):
            ld.make_blocks(["x"], 0)


class TestGreedyCompressPass(unittest.TestCase):
    def setUp(self):
        self.ts_rx = ld.DEFAULT_TS_RX

    def test_no_lines(self):
        result = ld.greedy_compress_pass([], self.ts_rx, 1)
        self.assertEqual(result, [])

    def test_dedup_identical_except_timestamps(self):
        lines = [
            '{"time":"2025-09-25T04:36:33Z","msg":"foo"}',
            '{"time":"2025-09-25T04:36:34Z","msg":"foo"}',
        ]
        result = ld.greedy_compress_pass(lines, self.ts_rx, 1)
        self.assertEqual(len(result), 1)
        self.assertIn('"foo"', result[0])

    def test_different_messages_not_deduped(self):
        lines = [
            '{"time":"2025-09-25T04:36:33Z","msg":"foo"}',
            '{"time":"2025-09-25T04:36:33Z","msg":"bar"}',
        ]
        result = ld.greedy_compress_pass(lines, self.ts_rx, 1)
        self.assertEqual(len(result), 2)


class TestGreedyCompressLines(unittest.TestCase):
    def setUp(self):
        self.ts_rx = ld.DEFAULT_TS_RX

    def test_env_disabled(self):
        """Should return input unchanged when LOG_TRIM is unset or invalid."""
        os.environ.pop("LOG_TRIM", None)
        text = "some logs\nmore logs"
        result = ld.greedy_compress_lines(text)
        self.assertEqual(result, text)

    def test_env_enabled(self):
        """Should dedup when LOG_TRIM is set."""
        os.environ["LOG_TRIM"] = "2"
        text = (
            '{"time":"2025-09-25T04:36:33Z","message":"TLS disabled."}\n'
            '{"time":"2025-09-25T04:36:34Z","message":"TLS disabled."}\n'
        )
        result = ld.greedy_compress_lines(text, self.ts_rx)
        # Expect only one entry remains after deduplication
        self.assertIn("TLS disabled", result)
        self.assertEqual(result.count("TLS disabled"), 1)

    def test_invalid_env(self):
        """Invalid LOG_TRIM should skip dedup."""
        os.environ["LOG_TRIM"] = "invalid"
        text = "hello world"
        result = ld.greedy_compress_lines(text)
        self.assertEqual(result, text)


class TestRealisticLogs(unittest.TestCase):
    """Integration-style tests using the provided log sequences."""

    def setUp(self):
        self.ts_rx = ld.DEFAULT_TS_RX
        os.environ["LOG_TRIM"] = "3"

    def test_dedup_kubernetes_events(self):
        sample = """
Error: Your service/namespace does not exist. Use kubectl to check.
{"level":"info","time":"2025-09-25T04:36:33Z","message":"TLS disabled."}
{"level":"info","time":"2025-09-25T04:36:33Z","message":"TLS disabled."}
{"level":"info","time":"2025-09-25T04:36:34Z","message":"TLS disabled."}
"""
        result = ld.greedy_compress_lines(sample.strip(), self.ts_rx)
        # Should remove duplicate lines differing only by timestamps
        self.assertIn("TLS disabled", result)
        self.assertEqual(result.count("TLS disabled"), 1)

    def test_k8s_container_events(self):
        sample = """
[90m2025-09-25T04:36:33Z[0m [32mINF[0m [1mcmd/geo/db.go:29[0m[36m >[0m [1mNew session successfull...[0m
[90m2025-09-25T04:36:34Z[0m [32mINF[0m [1mcmd/geo/db.go:29[0m[36m >[0m [1mNew session successfull...[0m
[90m2025-09-25T04:36:35Z[0m [32mINF[0m [1mcmd/geo/db.go:31[0m[36m >[0m [1mGenerating test data...[0m
"""
        result = ld.greedy_compress_lines(sample.strip(), self.ts_rx)
        # Should collapse identical log messages differing only by timestamp
        self.assertIn("New session successfull...", result)
        self.assertEqual(result.count("New session successfull..."), 1)
        # Should preserve distinct lines
        self.assertIn("Generating test data...", result)

    def test_multiblock_dedup(self):
        sample = "\n".join(
            [
                f'{{"level":"info","time":"2025-09-25T04:37:{i:02d}Z","message":"Tune: setGCPercent to 100"}}'
                for i in range(3)
            ]
        )
        result = ld.greedy_compress_lines(sample, self.ts_rx)
        self.assertIn("Tune: setGCPercent to 100", result)
        self.assertEqual(result.count("Tune: setGCPercent to 100"), 1)


if __name__ == "__main__":
    unittest.main()

