from .any_abs_gt_one import AnyAbsGreaterThanOneTask
from .has_all_tokens import HasAllTokensTask
from .has_pos_and_neg import HasPosAndNegTask
from .sign_of_second_place import SignOfSecondPlaceTask
from .single_digit_string_sum import SingleDigitStringSumTask
from .sign_of_winner import SignOfWinnerTask


# Central registry of available tasks. This module is standalone so that
# other modules (e.g., data generation) can import it without pulling in
# the package __init__ and causing circular imports.
TASK_REGISTRY = {
    "sign_of_winner": SignOfWinnerTask(),
    "sign_of_second_place": SignOfSecondPlaceTask(),
    "has_pos_and_neg": HasPosAndNegTask(),
    "has_all_tokens": HasAllTokensTask(),
    "any_abs_gt_one": AnyAbsGreaterThanOneTask(),
    "single_digit_string_sum": SingleDigitStringSumTask(),
}

__all__ = ["TASK_REGISTRY"]
