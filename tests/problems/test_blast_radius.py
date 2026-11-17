import unittest
from unittest import mock

from aiopslab.utils.blast_radius import (
    ERROR_STATUS,
    SpanMetricsQuery,
    blast_radius,
    rate_drops,
    score_submission,
    successful_rates,
)


class SuccessfulRatesTest(unittest.TestCase):

    def test_error_spans_do_not_count_as_success(self):
        samples = [
            ("frontend", "STATUS_CODE_UNSET", 300.0),
            ("frontend", ERROR_STATUS, 60.0),
            ("cart", "STATUS_CODE_OK", 120.0),
        ]
        rates = successful_rates(samples, seconds=60)
        self.assertAlmostEqual(rates["frontend"], 5.0)
        self.assertAlmostEqual(rates["cart"], 2.0)

    def test_service_with_only_errors_has_zero_rate(self):
        rates = successful_rates([("checkout", ERROR_STATUS, 50.0)], seconds=10)
        self.assertEqual(rates, {"checkout": 0.0})

    def test_rejects_empty_window(self):
        with self.assertRaises(ValueError):
            successful_rates([], seconds=0)


class RateDropsTest(unittest.TestCase):

    def test_drop_is_clipped_and_quiet_services_are_skipped(self):
        before = {"frontend": 10.0, "cart": 4.0, "email": 0.01, "ad": 2.0}
        during = {"frontend": 6.0, "cart": 5.0, "email": 0.0}
        drops = rate_drops(before, during, min_rate=0.05)
        self.assertAlmostEqual(drops["frontend"], 0.4)
        self.assertEqual(drops["cart"], 0.0)  # more traffic is not a negative drop
        self.assertEqual(drops["ad"], 1.0)  # disappeared during the fault
        self.assertNotIn("email", drops)


class BlastRadiusTest(unittest.TestCase):

    def test_threshold_is_inclusive_and_failed_service_is_left_out(self):
        drops = {"checkout": 1.0, "frontend": 0.2, "cart": 0.19, "payment": 0.9}
        self.assertEqual(blast_radius(drops, "checkout", 0.2), {"frontend", "payment"})

    def test_rejects_bad_threshold(self):
        with self.assertRaises(ValueError):
            blast_radius({}, "x", 0.0)


class ScoreSubmissionTest(unittest.TestCase):

    def test_perfect_answer_naming_the_failed_service(self):
        s = score_submission(["Checkout", "frontend", "payment"], {"frontend", "payment"}, "checkout")
        self.assertEqual(s["f1"], 1.0)
        self.assertTrue(s["exact_match"])
        self.assertTrue(s["names_failed_service"])

    def test_partial_answer(self):
        s = score_submission(["frontend", "email"], {"frontend", "payment", "shipping"}, "checkout")
        self.assertAlmostEqual(s["precision"], 0.5)
        self.assertAlmostEqual(s["recall"], 1 / 3)
        self.assertAlmostEqual(s["f1"], 0.4)
        self.assertAlmostEqual(s["jaccard"], 0.25)
        self.assertFalse(s["exact_match"])
        self.assertFalse(s["names_failed_service"])

    def test_both_empty_is_correct_and_empty_answer_to_real_set_is_not(self):
        self.assertEqual(score_submission([], set(), "cart")["f1"], 1.0)
        s = score_submission([], {"frontend"}, "cart")
        self.assertEqual((s["precision"], s["recall"], s["f1"]), (0.0, 0.0, 0.0))


class SpanMetricsQueryTest(unittest.TestCase):

    def test_builds_server_span_query_and_parses_rows(self):
        body = {
            "status": "success",
            "data": {
                "result": [
                    {"metric": {"service_name": "cart", "status_code": "STATUS_CODE_UNSET"}, "value": [0, "90"]},
                    {"metric": {"service_name": "cart", "status_code": ERROR_STATUS}, "value": [0, "30"]},
                ]
            },
        }
        resp = mock.Mock()
        resp.json.return_value = body
        resp.raise_for_status.return_value = None
        with mock.patch("aiopslab.utils.blast_radius.requests.get", return_value=resp) as get:
            rates = SpanMetricsQuery("http://prom:9090/").success_rates(1000.0, 1060.0)
        params = get.call_args.kwargs["params"]
        self.assertIn('span_kind=~"SPAN_KIND_SERVER|SPAN_KIND_CONSUMER"', params["query"])
        self.assertIn("[60s]", params["query"])
        self.assertEqual(params["time"], 1060.0)
        self.assertEqual(get.call_args.args[0], "http://prom:9090/api/v1/query")
        self.assertAlmostEqual(rates["cart"], 1.5)

    def test_rate_uses_the_same_rounded_window_as_the_query(self):
        body = {"status": "success", "data": {"result": [
            {"metric": {"service_name": "cart", "status_code": "STATUS_CODE_UNSET"}, "value": [0, "150"]}]}}
        resp = mock.Mock()
        resp.json.return_value = body
        resp.raise_for_status.return_value = None
        with mock.patch("aiopslab.utils.blast_radius.requests.get", return_value=resp) as get:
            rates = SpanMetricsQuery("http://prom").success_rates(1000.0, 1150.4)
        self.assertIn("[150s]", get.call_args.kwargs["params"]["query"])
        self.assertAlmostEqual(rates["cart"], 1.0)

    def test_failed_query_raises(self):
        resp = mock.Mock()
        resp.json.return_value = {"status": "error", "error": "bad"}
        resp.raise_for_status.return_value = None
        with mock.patch("aiopslab.utils.blast_radius.requests.get", return_value=resp):
            with self.assertRaises(RuntimeError):
                SpanMetricsQuery("http://prom").window_counts(0, 60)


if __name__ == "__main__":
    unittest.main()
