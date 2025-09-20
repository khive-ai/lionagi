from enum import Enum

from .action import ActionRequestRule
from .boolean import BooleanRule
from .choice import ChoiceRule
from .mapping import MappingRule
from .model import ModelRule
from .number import NumberRule
from .string import StringRule


class DEFAULT_RULES(Enum):
    CHOICE = ChoiceRule
    MAPPING = MappingRule
    MODEL = ModelRule
    NUMBER = NumberRule
    BOOL = BooleanRule
    STR = StringRule
    ACTION = ActionRequestRule
