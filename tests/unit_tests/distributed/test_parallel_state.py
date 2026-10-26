# Copyright (c) 2026, Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import socket
import subprocess
import sys
import unittest
from pathlib import Path

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from fsdp_turbo.distributed.parallel_state import ParallelState, Singleton

WORLD_SIZE = 4
WORKER_MARKER_ENV = 'FSDPTURBO_TEST_DIST_WORKER'
REPO_ROOT = Path(__file__).resolve().parents[3]


def _in_dist_worker():
    return WORKER_MARKER_ENV in os.environ or 'WORLD_SIZE' in os.environ


def _find_free_port():
    with socket.socket() as free_port_socket:
        free_port_socket.bind(('127.0.0.1', 0))
        return free_port_socket.getsockname()[1]


def _fresh_parallel_state():
    Singleton._instances.pop(ParallelState, None)
    return ParallelState()


class UpdateDeviceMeshMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _in_dist_worker():
            dist.init_process_group(backend='gloo')
            from fsdp_turbo.distributed import parallel_state

            cls._parallel_state_module = parallel_state
            cls._original_get_accelerator_name = parallel_state.get_accelerator_name
            # CPU/gloo workers have no NPU/CUDA accelerator.
            parallel_state.get_accelerator_name = lambda: 'cpu'
        else:
            command = [
                sys.executable,
                '-m',
                'torch.distributed.run',
                '--master_addr=127.0.0.1',
                f'--master_port={_find_free_port()}',
                f'--nproc_per_node={WORLD_SIZE}',
                '-m',
                'pytest',
                '-q',
                str(Path(__file__).resolve()),
                '-k',
                cls.__name__,
            ]
            environment = dict(os.environ, **{WORKER_MARKER_ENV: '1'})
            python_path = [str(REPO_ROOT)]
            if environment.get('PYTHONPATH'):
                python_path.append(environment['PYTHONPATH'])
            environment['PYTHONPATH'] = os.pathsep.join(python_path)
            result = subprocess.run(command, env=environment, check=False, timeout=600)
            if result.returncode != 0:
                raise AssertionError(f'Distributed worker run failed with exit code {result.returncode}.')

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
            cls._parallel_state_module.get_accelerator_name = cls._original_get_accelerator_name

    def setUp(self):
        if not dist.is_initialized():
            self.skipTest('Assertions run inside the distributed worker subprocess.')

    def _strided_group(self):
        """Return the strided ([0, 2] or [1, 3]) process group that contains this rank."""
        groups = [dist.new_group(ranks=[0, 2]), dist.new_group(ranks=[1, 3])]
        return groups[0] if dist.get_rank() in (0, 2) else groups[1]

    def test_device_mesh_passthrough_and_new_name(self):
        state = _fresh_parallel_state()
        mesh_2d = init_device_mesh('cpu', (2, 2), mesh_dim_names=('pipeline', 'tensor'))
        state.update_device_mesh_map({'pp': mesh_2d})

        self.assertIs(state.device_mesh_map['pp'], mesh_2d)
        self.assertIs(state.get_pp_device_mesh(), mesh_2d)
        self.assertEqual(len(state.get_pp_group()), 2)
        self.assertTrue(state.is_pp_enable())

    def test_empty_map_is_noop(self):
        state = _fresh_parallel_state()
        before = dict(state.device_mesh_map)
        state.update_device_mesh_map({})
        self.assertEqual(before, state.device_mesh_map)

    def test_invalid_mesh_type_raises(self):
        state = _fresh_parallel_state()
        for invalid in ('tp', None, [0, 1]):
            with self.assertRaisesRegex(TypeError, 'tp'):
                state.update_device_mesh_map({'tp': invalid})

    def test_partial_update_only_touches_given_names(self):
        state = _fresh_parallel_state()
        before = dict(state.device_mesh_map)
        state.update_device_mesh_map({'tp': self._strided_group()})

        self.assertIsNot(state.device_mesh_map['tp'], before['tp'])
        for mesh_name, mesh in before.items():
            if mesh_name != 'tp':
                self.assertIs(state.device_mesh_map[mesh_name], mesh)

    def test_process_group_is_wrapped_and_reused(self):
        state = _fresh_parallel_state()
        group = self._strided_group()
        state.update_device_mesh_map({'tp': group})

        mesh = state.device_mesh_map['tp']
        self.assertIsInstance(mesh, DeviceMesh)
        self.assertEqual(mesh.mesh_dim_names, ('tp',))
        self.assertIs(state.get_tp_group(), group)
        self.assertEqual(state.get_tp_group_size(), 2)
        self.assertEqual(state.get_tp_rank(), dist.get_rank() // 2)
        self.assertTrue(state.is_tp_enable())

    def test_state_printable_after_update(self):
        state = _fresh_parallel_state()
        state.update_device_mesh_map({'tp': self._strided_group()})
        self.assertIn('tp', str(state))


if __name__ == '__main__':
    unittest.main()
