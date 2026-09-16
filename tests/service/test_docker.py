import unittest

from aiopslab.service.dock import Docker


class DockerExecTimeoutTest(unittest.TestCase):
    def test_exec_command_times_out(self):
        docker = object.__new__(Docker)

        with self.assertRaises(RuntimeError):
            docker.exec_command("sleep 1", timeout=0.01)
