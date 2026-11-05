"""Unit tests for import_manual_runs_json_v2.py per-turn row extraction."""

from manual_runs.scripts.vllm.import_manual_runs_json_v2 import extract_per_turn_rows

BASE_ROW = {
    "run": "H200-test-model-8",
    "measured rps": 5.0,
    "output_tok/sec": 1000.0,
    "total_tok/sec": 11000.0,
    "prompt_token_count_p99": 1024.0,
    "output_token_count_p99": 128.0,
    "turn_index": "",
}


def _request(turn_index, output_tokens=100, prompt_tokens=1000):
    """Build a guidellm request record for the given turn."""
    return {
        "info": {"turn_index": turn_index},
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "time_to_first_token_ms": 50.0,
        "inter_token_latency_ms": 10.0,
        "time_per_output_token_ms": 10.0,
        "request_latency": 1.0,
        "output_tokens_per_second": 100.0,
        "tokens_per_second": 1100.0,
    }


def _benchmark(successful, errored=None, duration=10.0):
    """Build a guidellm benchmark section with an optional run duration."""
    benchmark = {
        "requests": {"successful": successful, "errored": errored or []},
    }
    if duration is not None:
        benchmark["scheduler_metrics"] = {
            "start_time": 100.0,
            "end_time": 100.0 + duration,
        }
    return benchmark


class TestExtractPerTurnRows:
    """Test the extract_per_turn_rows function."""

    def test_multi_turn_emits_one_row_per_turn(self):
        """Two distinct turn indices produce one row per turn."""
        rows = extract_per_turn_rows(
            _benchmark(
                [_request(0) for _ in range(5)] + [_request(1) for _ in range(5)]
            ),
            dict(BASE_ROW),
        )
        assert [row["turn_index"] for row in rows] == [0, 1]

    def test_throughput_is_turn_totals_over_measurement_interval(self):
        """Per-turn throughput is turn tokens / run duration, not per-request rate.

        5 requests x 100 output tokens over a 10s interval is 50 output tok/s
        (and 550 total tok/s with 1000 prompt tokens each), not the 100 tok/s
        per-request rate.
        """
        rows = extract_per_turn_rows(
            _benchmark(
                [_request(0) for _ in range(5)] + [_request(1) for _ in range(5)]
            ),
            dict(BASE_ROW),
        )
        for row in rows:
            assert row["output_tok/sec"] == 50.0
            assert row["total_tok/sec"] == 550.0

    def test_throughput_sums_to_aggregate_across_turns(self):
        """Per-turn throughput values sum to the run's total throughput."""
        rows = extract_per_turn_rows(
            _benchmark(
                [_request(0, output_tokens=60) for _ in range(5)]
                + [_request(1, output_tokens=40) for _ in range(10)]
            ),
            dict(BASE_ROW),
        )
        # Turn 0: 5*60/10 = 30, turn 1: 10*40/10 = 40, total = 70 tok/s.
        assert sum(row["output_tok/sec"] for row in rows) == 70.0

    def test_token_count_p99_computed_per_turn(self):
        """Token count p99 comes from the turn's requests, not the aggregate."""
        rows = extract_per_turn_rows(
            _benchmark(
                [_request(0, output_tokens=100) for _ in range(5)]
                + [_request(1, output_tokens=200) for _ in range(5)]
            ),
            dict(BASE_ROW),
        )
        assert rows[0]["output_token_count_p99"] == 100.0
        assert rows[1]["output_token_count_p99"] == 200.0

    def test_measured_rps_cleared_on_per_turn_rows(self):
        """measured rps is run-level context and is not inherited per turn."""
        rows = extract_per_turn_rows(
            _benchmark([_request(0), _request(1)]), dict(BASE_ROW)
        )
        assert all(row["measured rps"] is None for row in rows)

    def test_errored_requests_counted_per_turn(self):
        """Errored requests are counted per turn and participate in discovery."""
        rows = extract_per_turn_rows(
            _benchmark(
                [_request(0) for _ in range(4)],
                errored=[_request(0), _request(1)],
            ),
            dict(BASE_ROW),
        )
        by_turn = {row["turn_index"]: row for row in rows}
        assert by_turn[0]["successful_requests"] == 4
        assert by_turn[0]["errored_requests"] == 1
        assert by_turn[1]["errored_requests"] == 1
        assert by_turn[1]["successful_requests"] == 0

    def test_single_turn_emits_no_rows(self):
        """A single distinct turn index produces no per-turn rows."""
        rows = extract_per_turn_rows(
            _benchmark([_request(0) for _ in range(5)]), dict(BASE_ROW)
        )
        assert rows == []

    def test_missing_turn_index_emits_no_rows(self):
        """Requests without turn_index data produce no per-turn rows."""
        requests = [
            {k: v for k, v in _request(0).items() if k != "info"} for _ in range(5)
        ]
        rows = extract_per_turn_rows(_benchmark(requests), dict(BASE_ROW))
        assert rows == []

    def test_missing_duration_leaves_throughput_empty(self):
        """Without a scheduler measurement interval, throughput is not computed."""
        rows = extract_per_turn_rows(
            _benchmark([_request(0), _request(1)], duration=None), dict(BASE_ROW)
        )
        assert len(rows) == 2
        assert all(row["output_tok/sec"] is None for row in rows)
        assert all(row["total_tok/sec"] is None for row in rows)
