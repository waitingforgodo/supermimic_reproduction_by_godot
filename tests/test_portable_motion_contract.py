"""A relocated/canonicalized archive is equivalent; changed physics data is not."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from g1_supermimic.configuration import digest


class PortableMotionTests(unittest.TestCase):
    def test_content_hash_ignores_zip_encoding_and_source_location_only(self):
        values=dict(fps=np.array([50.],dtype=np.float32),joint_names=np.array(['joint']),
            body_names=np.array(['pelvis']),joint_pos=np.zeros((3,1)),joint_vel=np.zeros((3,1)),
            body_pos_w=np.zeros((3,1,3)),body_quat_w=np.tile([1.,0.,0.,0.],(3,1,1)),
            body_lin_vel_w=np.zeros((3,1,3)),body_ang_vel_w=np.zeros((3,1,3)))
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a.npz',Path(folder)/'b.npz'
            np.savez(a,**values,source='/old/machine/input.csv')
            np.savez_compressed(b,**values,source='input.csv')
            self.assertEqual(digest(a),digest(b))
            values['joint_pos'][1,0]=.01
            np.savez_compressed(b,**values,source='input.csv')
            self.assertNotEqual(digest(a),digest(b))
