"""Reuse the original episode code with only request/response bindings for ID 8."""
from types import FunctionType
from research.pose_language_contracts import instruction_fields,validate_response


def language_episode(spec):
    # Import only after SimulationApp. The original module/globals remain intact.
    from research import static_obstacle_navigation as baseline
    return bind_episode(baseline,spec)


def bind_episode(baseline,spec):
    original=baseline.request

    def request(*args,**kwargs):
        value=original(*args,**kwargs)
        value.update(instruction_fields(spec))
        return value

    scope=dict(baseline.episode.__globals__,request=request,validate_response=validate_response)
    function=FunctionType(baseline.episode.__code__,scope,'pose_language_episode')
    assert function.__code__ is baseline.episode.__code__
    return function
