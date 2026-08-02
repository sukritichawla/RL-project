from enum import IntEnum


class Action(IntEnum):
    PUMP_ON = 0
    PUMP_OFF = 1

    INCREASE_PUMP_SPEED = 2
    DECREASE_PUMP_SPEED = 3

    OPEN_VALVE = 4
    CLOSE_VALVE = 5


ACTION_NAMES = {
    Action.PUMP_ON: "Pump ON",
    Action.PUMP_OFF: "Pump OFF",
    Action.INCREASE_PUMP_SPEED: "Increase Pump Speed",
    Action.DECREASE_PUMP_SPEED: "Decrease Pump Speed",
    Action.OPEN_VALVE: "Open Valve",
    Action.CLOSE_VALVE: "Close Valve"
}