"""No provider calls, dependencies, cluster work or generated-score fixtures."""
import unittest
from dataclasses import asdict
import json
from engine.mf_gp.v3.cost import CostRecord, Workload, action_cost, completion_guard, feasible
from engine.mf_gp.v3.context import BirthContext, common_scale


class V3FoundationTests(unittest.TestCase):
    def test_failure_cost_is_spent_but_not_training_sample(self):
        rows = [CostRecord("p", .2, s, 3, 5, 100, 7)
                for s in ("completed", "failed", "timeout", "interrupted")]
        self.assertEqual(sum(r.spent for r in rows), 460)
        self.assertEqual([r.completed_process_sample for r in rows], [100, None, None, None])
        self.assertEqual([r.completion_lower_bound for r in rows], [None, None, 100, None])
        self.assertEqual(CostRecord(**json.loads(json.dumps(asdict(rows[0])))), rows[0])

    def test_same_scope_and_sunk_generation(self):
        w = Workload(10, 100, 2, 20, 5)
        self.assertEqual(action_cost(w), 235)
        self.assertEqual(action_cost(w, generation_remaining=30, probe_remaining=10), 275)
        # Full restart includes all full steps, not just additional 80%.
        full = Workload(10, 500, 2, 20, 5)
        self.assertEqual(action_cost(full), 1035)

    def test_expensive_program_does_not_disable_cheap_program(self):
        self.assertFalse(feasible(100, completion_guard(1000)))
        self.assertTrue(feasible(100, completion_guard(10), required_full_guard=60))
        self.assertFalse(feasible(100, completion_guard(10), required_full_guard=90))
        self.assertEqual(completion_guard(10, [100]), 150)

    def test_scale_updates_without_rewriting_birth_context(self):
        raw = BirthContext(-.56, 0., .1, 2, 3, .2, False, 1)
        old = raw.features(*common_scale([-.56]))
        restored = BirthContext(**json.loads(json.dumps(asdict(raw))))
        transform = common_scale([-.56, -2.65])
        self.assertGreater(transform[1], 1.)
        self.assertEqual(raw.features(*transform), restored.features(*transform))
        self.assertNotEqual(old, raw.features(*transform))
        self.assertEqual(common_scale([-2.65, -.56]), transform)

    def test_invalid_values_fail_closed(self):
        for bad in (float("nan"), float("inf"), -1., True):
            with self.assertRaises(ValueError):
                CostRecord("p", .2, "completed", process=bad)
        with self.assertRaises(ValueError): common_scale([float("nan")])
        with self.assertRaises(ValueError): Workload(0, -1, 1, 0, 0)

if __name__ == "__main__":
    unittest.main()
