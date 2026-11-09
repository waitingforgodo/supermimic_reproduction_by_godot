"""A conservative policy update must preserve buffers/learners and invalidate stale scores."""
from copy import deepcopy
import unittest
import torch
from g1_supermimic.policy_backtrack import blend_policy


class BacktrackTests(unittest.TestCase):
    def checkpoints(self):
        base = dict(config={'optimizations': {'motion_anchor_ori_b': True}},
                    policy={'weight': torch.tensor([1.,2.]), 'scale': torch.tensor([3.]),
                            'count': torch.tensor(50)}, policy_opt={'state': {0: {'step': 4}}},
                    wm={'weight': torch.tensor([7.])}, critic={'weight': torch.tensor([8.])},
                    bin_visits=torch.tensor([4,6]), iteration=20,
                    selection={'score': [1.,1.,.9]})
        candidate = deepcopy(base)
        candidate['policy']['weight'] = torch.tensor([3.,4.])
        return base, candidate

    def test_global_blend_only_changes_policy_parameters_and_resets_its_optimizer(self):
        base, candidate = self.checkpoints()
        result = blend_policy(base,candidate,{'weight'},.5)
        torch.testing.assert_close(result['policy']['weight'], torch.tensor([2.,3.]))
        for key in ('scale','count'):
            self.assertTrue(torch.equal(result['policy'][key],candidate['policy'][key]))
        for key in ('wm','critic','bin_visits'):
            self.assertIs(result[key],candidate[key])
        self.assertFalse(result['policy_opt']['state'])
        self.assertTrue(candidate['policy_opt']['state'])
        self.assertEqual(result['iteration'],candidate['iteration'])
        self.assertNotIn('selection',result)

    def test_different_normalization_or_feature_semantics_are_rejected(self):
        base, candidate = self.checkpoints()
        candidate['policy']['scale'].add_(1)
        with self.assertRaisesRegex(ValueError,'Normalization'):
            blend_policy(base,candidate,{'weight'},.5)
        base, candidate = self.checkpoints()
        candidate['config']['optimizations']['motion_anchor_ori_b'] = False
        with self.assertRaisesRegex(ValueError,'semantics'):
            blend_policy(base,candidate,{'weight'},.5)

    def test_invalid_alpha_or_nonfinite_result_is_rejected(self):
        base, candidate = self.checkpoints()
        for value in (0,1,-1,True,float('nan')):
            with self.subTest(alpha=value),self.assertRaises(ValueError):
                blend_policy(base,candidate,{'weight'},value)
        candidate['policy']['weight'].fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            blend_policy(base,candidate,{'weight'},.5)
