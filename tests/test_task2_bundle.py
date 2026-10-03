"""Application-side routing contracts without requiring ONNX installation."""
import unittest
import numpy as np
from exports.task2_bundle import route

class Session:
    def __init__(self, value):
        self.value = value
        self.calls = []
    def run(self, outputs, inputs):
        x = inputs['image']
        self.calls.append(len(x))
        return [np.full_like(x, self.value)]

class RoutingTest(unittest.TestCase):
    def test_mixed_batch_dispatch_and_identity(self):
        sessions = {k: Session(v) for k,v in [('salt',.1),('blur',.2),('occlusion',.3)]}
        x = np.full((5,3,128,128), .75, np.float32)
        labels = np.array([3,0,1,3,2], dtype=np.int64)
        y,_ = route(sessions,x,labels)
        np.testing.assert_array_equal(y[1],x[1])
        np.testing.assert_allclose(y[:,0,0,0],[.3,.75,.1,.3,.2])
        self.assertEqual(sessions['occlusion'].calls,[2])
        np.testing.assert_array_equal(x,np.full_like(x,.75))
    def test_clean_does_not_call_experts(self):
        x = np.zeros((2,3,128,128),np.float32)
        y,_ = route({},x,np.zeros(2,dtype=np.int64))
        np.testing.assert_array_equal(y,x)
    def test_reject_invalid_contract(self):
        x = np.zeros((1,3,128,128),np.float32)
        for labels in [np.array([4]),np.array([-1]),np.array([1.0])]:
            with self.assertRaises(ValueError): route({},x,labels)
        with self.assertRaises(ValueError): route({},x.astype(np.float64),np.array([0]))

if __name__ == '__main__': unittest.main()
