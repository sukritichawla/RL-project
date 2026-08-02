def calculate_reward(state):
    """
    Reward Function

    Positive reward:
        • Pressure maintained
        • Demand satisfied
        • High tank level

    Negative reward:
        • Low pressure
        • Empty tank
        • Unnecessary pump usage
    """

    reward = 0

    # Good pressure
    if 40 <= state.pressure <= 70:
        reward += 20
    else:
        reward -= 20

    # Tank level
    if state.tank_level > 30:
        reward += 15
    else:
        reward -= 15

    # Demand satisfaction
    if state.demand <= state.tank_level:
        reward += 25
    else:
        reward -= 30

    # Pump penalty
    if state.pump_status == 1:
        reward -= 5

    return reward