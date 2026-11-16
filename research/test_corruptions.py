import unittest
import numpy as np
from research.corruptions import CASES,mask,gap_summary

class CorruptionTests(unittest.TestCase):
    def test_determinism_and_template_is_not_mutated(self):
        template=np.array([0,0,1,1,1],dtype=bool)
        before=template.copy()
        for case in CASES:
            a=mask(case,1000,template,101)
            self.assertEqual(a.dtype,np.dtype(bool))
            np.testing.assert_array_equal(a,mask(case,1000,template,101))
        np.testing.assert_array_equal(template,before)

    def test_stationary_rates_and_burst_lengths(self):
        for case,seconds in [('iid40',None),('burst2s40',2),('burst10s40',10)]:
            a=mask(case,1000000,[False,True],202)
            s=gap_summary(a)
            self.assertLess(abs(s['missing_fraction']-.4),.03)
            if seconds:
                self.assertLess(abs(s['mean_gap_s']-seconds),seconds*.1)
        self.assertFalse(mask('visible',10,[True],101).any())

    def test_target_rotation_preserves_complete_cycle(self):
        template=np.array([False,False,False,True,True])
        a=mask('target_like',1000,template,303)
        self.assertEqual(a.sum(),400)
        self.assertEqual(gap_summary(np.array([0,1,1,0],bool))['max_gap_s'],2/25)

if __name__=='__main__':
    unittest.main()
