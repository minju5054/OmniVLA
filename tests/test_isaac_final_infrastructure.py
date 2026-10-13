"""Deterministic infrastructure checks; no scientific model navigation."""
import ast
import json
import math
from pathlib import Path
import socket
import tempfile
import unittest
import numpy as np
from research.isaac_final_contracts import Client, History, Scheduler, action, integrate, local_goal
from research.isaac_final_common import GATES, verify_frozen, write_json

ROOT = Path(__file__).resolve().parents[1]


class AdapterTests(unittest.TestCase):
    def test_front_left_right(self):
        for y in (0.,2.,-2.):
            local, normalized = local_goal([0.,0.,0.], [5.,y,math.atan2(y,5)])
            np.testing.assert_allclose(local[:2], [5,y])
            np.testing.assert_allclose(normalized[:2], [50,y*10])
            self.assertAlmostEqual(normalized[2]**2+normalized[3]**2, 1.)

    def test_rotated_translated_observation_pose(self):
        local, norm = local_goal([2.,3.,math.pi/2], [2.,8.,math.pi/2])
        np.testing.assert_allclose(local, [5,0,0], atol=1e-14)
        np.testing.assert_allclose(norm, [50,0,1,0], atol=1e-13)

    def test_observation_anchor_and_clamp(self):
        old = local_goal([0,0,0], [100,100,math.pi])[1]
        new = local_goal([1,2,.5], [100,100,math.pi])[1]
        self.assertNotEqual(old,new)
        self.assertAlmostEqual(math.hypot(*old[:2]), 300)

    def test_pose_adapter_matches_upstream_compass_mapping(self):
        from research.edge_contracts import goal
        for robot, target in [([0,0,0],[5,2,.3]), ([2,3,.7],[-8,17,-.2]),
                              ([0,0,-.4],[100,100,.9])]:
            expected = goal(target[0]-robot[0], target[1]-robot[1],
                            90-math.degrees(robot[2]), 90-math.degrees(target[2]))
            np.testing.assert_allclose(local_goal(robot,target)[1], expected, atol=1e-12)

    def test_explicit_euler_and_zero_hold(self):
        self.assertEqual(integrate([0,0,0],[0,0]), [0,0,0])
        np.testing.assert_allclose(integrate([0,0,math.pi/2],[.3,.3]), [0,.005,math.pi/2+.005], atol=1e-14)

    def test_native_shape_finiteness_and_immutability(self):
        a = action(np.ones((1,8,4), dtype=np.float32))
        with self.assertRaises(ValueError):
            a.setflags(write=True)
        for invalid in (np.ones((8,4),dtype=np.float32), np.ones((1,8,4)),
                        np.full((1,8,4),np.nan,dtype=np.float32)):
            with self.assertRaises(ValueError):
                action(invalid)

    def test_history_independent_ids_order_and_no_future(self):
        h = History()
        for i in range(7):
            h.append(dict(frame_id=i, sim_time=i/3, path=f'{i}.png'))
        self.assertEqual([f['frame_id'] for f in h.snapshot(2.)], list(range(1,7)))
        with self.assertRaises(ValueError):
            h.snapshot(1.5)
        with self.assertRaises(ValueError):
            h.append(dict(frame_id=6,sim_time=7/3))
        with self.assertRaises(ValueError):
            h.append(dict(frame_id=7,sim_time=3.))

    def test_scheduler_busy_slots_hold_and_no_catchup(self):
        s = Scheduler()
        self.assertEqual(s.command, [0,0])
        self.assertEqual(s.slot(0), 'SEND')
        s.sent(0)
        self.assertEqual(s.slot(20), 'MISSED_INFERENCE_SLOT')
        self.assertEqual(s.slot(21), 'NONE')
        self.assertEqual(s.command, [0,0])
        with self.assertRaises(RuntimeError):
            s.sent(1)
        with self.assertRaises(ValueError):
            s.ready(1,[.1,.2])
        s.ready(0,[.1,.2])
        self.assertEqual(s.command, [.1,.2])
        self.assertEqual(s.slot(22), 'NONE')
        self.assertEqual(s.slot(40), 'SEND')
        s.sent(1)
        self.assertEqual(s.command, [.1,.2])
        self.assertEqual(s.missed,1)

    def test_ipc_fragmentation_single_outstanding_and_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp)/'ipc.sock')
            server = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            server.bind(path)
            server.listen(1)
            client = Client(path)
            peer, _ = server.accept()
            try:
                client.send(dict(request_id=7,observation_id=12))
                peer.recv(1024)
                with self.assertRaises(RuntimeError):
                    client.send(dict(request_id=8,observation_id=13))
                peer.sendall(b'{"request_id":7,')
                self.assertIsNone(client.poll())
                peer.sendall(b'"observation_id":12}\n')
                self.assertEqual(client.poll()['request_id'],7)
                client.send(dict(request_id=8,observation_id=13))
                peer.recv(1024)
                peer.sendall(b'{"request_id":8,"observation_id":14}\n')
                with self.assertRaises(ValueError):
                    client.poll()
            finally:
                peer.close();client.close();server.close()

    def test_durable_results_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'evidence.json'
            write_json(p, {'status':'pre-close'})
            with self.assertRaises(FileExistsError):
                write_json(p, {'status':'replacement'})

    def test_scientific_guard_rejects_unqualified_or_changed_code(self):
        with self.assertRaises(ValueError):
            verify_frozen({})
        config = dict(infrastructure_gates={key:dict(status='PASS') for key in GATES},
                      implementation_sha256={'scripts/isaac6_python.sh':'incorrect'})
        with self.assertRaisesRegex(ValueError, 'changed'):
            verify_frozen(config)

    def test_close_is_last_action_and_evidence_precedes_it(self):
        tree = ast.parse((ROOT/'research/isaac_final_run.py').read_text())
        main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        self.assertEqual(ast.unparse(main.body[-1]), 'app.close(exit_code=code)')
        self.assertNotIn("'fast_shutdown': False", ast.unparse(main))
        calls = [ast.unparse(n.func) for n in ast.walk(main) if isinstance(n,ast.Call)]
        self.assertIn('sync_tree', calls)


if __name__ == '__main__':
    unittest.main()
