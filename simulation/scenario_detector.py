def classify_from_grid(local_obs: dict, scenario_hint: str = None) -> str:
    """
    Uses local grid observations around the drone to classify scenario.
    In the browser sim, the scenario is known after placement — this
    function simulates the drone 'discovering' it over 2-3 steps.
    Returns: 'collapse' | 'wildfire' | 'flood'
    """
    # Simulate delayed discovery: drone learns scenario after 3 steps
    if local_obs.get("step_count", 0) < 3:
        return "unknown"
    return local_obs.get("scenario", scenario_hint)  # env reveals scenario after 3 steps
