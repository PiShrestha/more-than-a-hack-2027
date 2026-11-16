import unittest
import numpy as np
from research.cache_tools import configuration,probability,separate_periods,insert_context

FROZEN=dict(switch=4,die=8,revive=32,tcn_weight=.75,tcn_tags=['','_s1'])

class CacheTests(unittest.TestCase):
    def test_penalty_and_equivalent_mix_overrides_preserve_full_precision(self):
        full=np.array([[.50000000001,.25,.24999999999]],dtype=np.float64)
        cache={'ensemble':full}
        for override in ({'switch':2},{'tcn_weight':.75},{'tcn_tags':['_s1','']}):
            self.assertIs(probability(cache,configuration(FROZEN,override),FROZEN),full)
        self.assertEqual(FROZEN['tcn_tags'],['','_s1'])

    def test_explicit_member_and_weight_changes_use_the_requested_probabilities(self):
        cache=dict(lightgbm=np.array([[1.,0.,0.]]),tcn_base=np.array([[0.,1.,0.]]),tcn_s1=np.array([[0.,0.,1.]]))
        p=probability(cache,configuration(FROZEN,dict(tcn_weight=.5,tcn_tags=[''])),FROZEN)
        np.testing.assert_array_equal(p,[[.5,.5,0.]])
        np.testing.assert_array_equal(probability(cache,configuration(FROZEN,{'tcn_weight':0}),FROZEN),cache['lightgbm'])

    def test_invalid_settings_fail_instead_of_silently_changing_the_experiment(self):
        for override in ({'switch':-1},{'revive':float('nan')},{'tcn_weight':2},
                         {'tcn_tags':[]},{'tcn_tags':['','']},{'tcn_tags':['missing']},
                         {'switch_cost':1},{'temp':0},{'die':True},[],False,''):
            with self.subTest(override=override),self.assertRaises(ValueError):
                configuration(FROZEN,override)

    def test_halves_are_separated_and_ungraded_frames_removed(self):
        gt=np.array([3,0,0,1,1,3],np.int8)
        period=np.array([0,1,1,2,2,0],np.int8)
        a,b=separate_periods(gt,gt.copy(),period)
        np.testing.assert_array_equal(a,[0,0,3,1,1,3])
        np.testing.assert_array_equal(a,b)

    def test_inserted_context_preserves_every_play_feature_and_row_mapping(self):
        z=np.array([[10,11],[20,21],[30,31]],dtype=np.float32)
        period=np.array([1,1,2])
        context=[np.array([[-1,-2]]),np.array([[7,8],[9,10]]),np.empty((0,2))]
        expanded,index=insert_context(z,period,context)
        np.testing.assert_array_equal(index,[1,2,5])
        np.testing.assert_array_equal(expanded[index],z)
        np.testing.assert_array_equal(expanded[[0,3,4]],[[-1,-2],[7,8],[9,10]])
        with self.assertRaises(ValueError):
            insert_context(z,np.array([1,2,1]),context)

if __name__=='__main__':
    unittest.main()
