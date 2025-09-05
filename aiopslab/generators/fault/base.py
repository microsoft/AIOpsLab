# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Interface for fault injector classes.

NOTE: Each layer of injection like ApplicationFaultInjector, OSFaultInjector HardwareFaultInjector, etc.
are implemented as child classes of FaultInjector.
"""

import time


class FaultInjector:
    def __init__(self, testbed):
        self.testbed = testbed

    # Deprecated method
    def inject_fault(
        self,
        fault_type: str,
        fault_id: str,
        start_time: float,
        end_time: float,
        microservices: list[str] = None,
    ):
        """
        Base class to inject a fault into the specified microservices.

        Parameters:
        microservices (list[str]): list of microservices to inject the fault into.
        fault_type (str): Type of fault to inject.
        fault_id (str): Unique identifier for the fault.
        start_time (float): Time to start the fault injection (epoch time).
        end_time (float): Time to end the fault injection (epoch time).
        """
        current_time = time.time()
        if current_time < start_time:
            time.sleep(start_time - current_time)

        self._inject(microservices, fault_type)

    def _inject(
        self, fault_type: str, microservices: list[str] = None, duration: str = None
    ):
        if duration:
            self._invoke_method("inject", fault_type, microservices, duration)
        elif microservices:
            self._invoke_method("inject", fault_type, microservices)
        else:
            self._invoke_method("inject", fault_type)
        time.sleep(6)

    def _recover(
        self,
        fault_type: str,
        microservices: list[str] = None,
    ):
        if microservices and fault_type:
            self._invoke_method("recover", fault_type, microservices)
        elif fault_type:
            self._invoke_method("recover", fault_type)

    def _invoke_method(self, action_prefix, *args):
        """helper: injects/recovers faults based on name"""
        method_name = f"{action_prefix}_{args[0]}"
        method = getattr(self, method_name, None)
        if method:
            method(*args[1:])
        else:
            print(f"Unknown fault type: {args[0]}")
