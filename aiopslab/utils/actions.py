# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib
import inspect


def action(method):
    """
    Decorator to mark a method as an action.

    Args:
        method (function): The method to mark as an action.

    Returns:
        function: The decorated method.
    """
    method.is_action = True
    return method


def read(method):
    """
    Decorator to mark a method as a read action.

    Args:
        method (function): The method to mark as a read action.

    Returns:
        function: The decorated method.
    """
    method.is_action = True
    method.action_type = "read"
    return method


def write(method):
    """
    Decorator to mark a method as a write action.

    Args:
        method (function): The method to mark as a write action.

    Returns:
        function: The decorated method.
    """
    method.is_action = True
    method.action_type = "write"
    return method


def get_actions(task: str, subtype: str | None = None) -> dict:
    """
    Get all actions for the given task.
        key: action name
        value: docstring of the action

    Args:
        task (str): The name of the task.
        subtype (str): The subtype of the action (optional) (default: None).

    Returns:
        dict: A dictionary of actions for the given task.
    """
    class_name = task.title() + "Actions"
    module = importlib.import_module("aiopslab.orchestrator.actions." + task)
    class_obj = getattr(module, class_name)

    actions = {
        method: getattr(class_obj, method).__doc__.strip()
        for method in dir(class_obj)
        if callable(getattr(class_obj, method))
        and getattr(getattr(class_obj, method), "is_action", False)
    }

    if subtype:
        actions = {
            method: doc
            for method, doc in actions.items()
            if getattr(getattr(class_obj, method), "action_type", None) == subtype
        }

    return actions


def submitted_solution(actions, args: list, kwargs: dict):
    """
    Get the solution to store for a submit() call.

    A call that matches the task's submit() signature is resolved through it, so
    submit("Yes") and submit(has_anomaly="Yes") give the same solution. Any other
    call keeps the previous behavior.

    Args:
        actions: The task's actions class (or None if unknown).
        args (list): Positional arguments parsed from the call.
        kwargs (dict): Keyword arguments parsed from the call.

    Returns:
        The single submitted value, or a list of values.
    """
    try:
        bound = inspect.signature(actions.submit).bind(*args, **kwargs)
    except (AttributeError, TypeError, ValueError):
        return args[0] if len(args) == 1 else args
    values = list(bound.arguments.values())
    return values[0] if len(values) == 1 else values
