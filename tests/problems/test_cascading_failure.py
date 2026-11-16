import unittest
from unittest import mock

MODULE = "aiopslab.orchestrator.problems.cascading_failure.cascading_failure"


class _FakeApp:
    namespace = "astronomy-shop"
    helm_configs = {}

    def get_app_summary(self):
        return "Astronomy Shop"


def _patched():
    """Patch everything that would reach a live cluster.

    aiopslab.observer loads the kube config at import time, so that is patched
    too and the tests run on a machine without a cluster.
    """
    return [
        mock.patch("kubernetes.config.kube_config.load_kube_config"),
        mock.patch(f"{MODULE}.AstronomyShop", return_value=_FakeApp()),
        mock.patch(f"{MODULE}.KubeCtl"),
        mock.patch(f"{MODULE}.SymptomFaultInjector"),
        mock.patch("aiopslab.orchestrator.tasks.base.KubeCtl"),
    ]


class CascadingFailureTest(unittest.TestCase):

    def setUp(self):
        self.patches = [p.start() for p in _patched()]
        from aiopslab.orchestrator.problems.cascading_failure import cascading_failure as cf
        self.cf = cf

    def tearDown(self):
        mock.patch.stopall()

    def test_registry_has_three_hubs_for_each_task(self):
        from aiopslab.orchestrator.problems.registry import ProblemRegistry

        reg = ProblemRegistry()
        expected = {"1": "checkout", "2": "product-catalog", "3": "cart"}
        for task, cls in (
            ("detection", self.cf.CascadingFailureDetection),
            ("localization", self.cf.CascadingFailureLocalization),
            ("blast_radius", self.cf.CascadingFailureBlastRadius),
        ):
            for variant, hub in expected.items():
                prob = reg.get_problem_instance(f"astronomy_shop_cascading_failure-{task}-{variant}")
                self.assertIsInstance(prob, cls)
                self.assertEqual(prob.faulty_service, hub)
        self.assertEqual(len(reg.get_problem_ids("cascading_failure-blast_radius")), 3)

    def test_inject_fault_fails_every_pod_of_the_hub_and_records_baseline(self):
        prob = self.cf.CascadingFailureDetection("checkout")
        with mock.patch(f"{MODULE}.time.time", return_value=5000.0):
            prob.inject_fault()
        experiment, name = prob.injector.create_chaos_experiment.call_args.args
        spec = experiment["spec"]
        self.assertEqual(name, "cascading-failure")
        self.assertEqual(spec["action"], "pod-failure")
        self.assertEqual(spec["mode"], "all")
        self.assertEqual(spec["selector"]["labelSelectors"], {prob.POD_LABEL: "checkout"})
        self.assertEqual(prob.baseline_window, (4820.0, 5000.0))
        self.assertEqual(prob.injected_at, 5000.0)
        prob.recover_fault()
        prob.injector.delete_chaos_experiment.assert_called_once_with("cascading-failure")

    def test_measure_uses_settled_fault_window_and_baseline(self):
        prob = self.cf.CascadingFailureBlastRadius("cart")
        prob.baseline_window = (800.0, 1000.0)
        prob.injected_at = 1000.0
        query = mock.Mock()
        query.success_rates.side_effect = [
            {"frontend": 10.0, "cart": 4.0, "checkout": 2.0, "ad": 3.0},
            {"frontend": 7.0, "checkout": 0.5, "ad": 3.0},
        ]
        url_cm = mock.MagicMock()
        url_cm.__enter__.return_value = "http://prom"
        with mock.patch(f"{MODULE}.SpanMetricsQuery", return_value=query), \
                mock.patch(f"{MODULE}.prometheus_url", return_value=url_cm), \
                mock.patch(f"{MODULE}.time.time", return_value=1200.0), \
                mock.patch(f"{MODULE}.time.sleep") as sleep:
            truth, drops = prob.measure_blast_radius()
        sleep.assert_not_called()
        self.assertEqual(query.success_rates.call_args_list[0].args, (800.0, 1000.0))
        self.assertEqual(query.success_rates.call_args_list[1].args, (1030.0, 1200.0))
        self.assertEqual(truth, {"frontend", "checkout"})
        self.assertEqual(drops["cart"], 1.0)

    def test_measure_waits_for_enough_fault_data(self):
        prob = self.cf.CascadingFailureBlastRadius("cart")
        prob.baseline_window = (800.0, 1000.0)
        prob.injected_at = 1000.0
        query = mock.Mock()
        query.success_rates.return_value = {"frontend": 1.0}
        url_cm = mock.MagicMock()
        url_cm.__enter__.return_value = "http://prom"
        with mock.patch(f"{MODULE}.SpanMetricsQuery", return_value=query), \
                mock.patch(f"{MODULE}.prometheus_url", return_value=url_cm), \
                mock.patch(f"{MODULE}.time.time", return_value=1050.0), \
                mock.patch(f"{MODULE}.time.sleep") as sleep:
            prob.measure_blast_radius()
        sleep.assert_called_once_with(70.0)

    def test_measure_requires_injection_first(self):
        prob = self.cf.CascadingFailureBlastRadius("cart")
        with self.assertRaises(RuntimeError):
            prob.measure_blast_radius()

    def test_blast_radius_eval_scores_against_measured_set(self):
        prob = self.cf.CascadingFailureBlastRadius("checkout")
        measured = ({"frontend", "payment"}, {"checkout": 1.0, "frontend": 0.5, "payment": 0.9})
        with mock.patch.object(prob, "measure_blast_radius", return_value=measured):
            res = prob.eval(["checkout", "frontend", "email"], trace=[], duration=12.0)
        self.assertEqual(res["measured_blast_radius"], ["frontend", "payment"])
        self.assertAlmostEqual(res["Blast Radius Precision"], 0.5)
        self.assertAlmostEqual(res["Blast Radius Recall"], 0.5)
        self.assertTrue(res["Names Failed Service"])
        self.assertFalse(res["success"])
        self.assertEqual(res["TTL"], 12.0)

    def test_blast_radius_description_states_threshold(self):
        prob = self.cf.CascadingFailureBlastRadius("checkout")
        self.assertIn("20% or more", prob.get_task_description())

    def test_localization_eval_matches_hub(self):
        prob = self.cf.CascadingFailureLocalization("product-catalog")
        res = prob.eval(["product-catalog"], trace=[], duration=3.0)
        self.assertTrue(res["success"])
        self.assertEqual(res["Localization Accuracy"], 100.0)


if __name__ == "__main__":
    unittest.main()
