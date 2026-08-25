from __future__ import annotations

from typing import Sequence, Tuple

Word = Tuple[str, ...]


def is_inverse_token(token: str) -> bool:
    return token.endswith("^-1")


def base_token(token: str) -> str:
    return token[:-3] if is_inverse_token(token) else token


def invert_token(token: str) -> str:
    return token[:-3] if is_inverse_token(token) else f"{token}^-1"


def inverse_word(word: Sequence[str]) -> Word:
    return tuple(invert_token(t) for t in reversed(tuple(word)))


def free_reduce(word: Sequence[str]) -> Word:
    stack: list[str] = []
    for token in word:
        if stack and invert_token(token) == stack[-1]:
            stack.pop()
        else:
            stack.append(token)
    return tuple(stack)


def commutator_word(left: str, right: str) -> Word:
    return (left, right, invert_token(left), invert_token(right))
