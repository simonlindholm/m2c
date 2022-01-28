import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar, Union

from .parse_file import Label
from .parse_instruction import (
    Argument,
    AsmAddressMode,
    AsmGlobalSymbol,
    AsmLiteral,
    BinOp,
    Instruction,
    InstructionMeta,
    JumpTarget,
    NaiveParsingArch,
    Register,
    parse_instruction,
)


BodyPart = Union[Instruction, Label]
Pattern = List[Tuple[BodyPart, bool]]


def make_pattern(*parts: str) -> Pattern:
    ret: Pattern = []
    for part in parts:
        optional = part.endswith("?")
        part = part.rstrip("?")
        if part.endswith(":"):
            ret.append((Label(part.strip(".:")), optional))
        else:
            ins = parse_instruction(part, InstructionMeta.missing(), NaiveParsingArch())
            ret.append((ins, optional))
    return ret


@dataclass
class Replacement:
    new_body: Sequence[BodyPart]
    num_consumed: int


@dataclass
class AsmMatch:
    body: List[BodyPart]
    unrelated: List[Instruction]
    regs: Dict[str, Register]
    literals: Dict[str, int]
    instructions: Dict[str, Instruction]

    def derived_instr(self, mnemonic: str, args: List[Argument]) -> Instruction:
        old_instr = next(part for part in self.body if isinstance(part, Instruction))
        return Instruction.derived(mnemonic, args, old_instr)

    def replace(self, *new_body: BodyPart) -> Replacement:
        return Replacement(new_body, len(self.body) + len(self.unrelated))


class AsmPattern(abc.ABC):
    @abc.abstractmethod
    def match(self, matcher: "AsmMatcher") -> Optional[Replacement]:
        ...


class SimpleAsmPattern(AsmPattern):
    @property
    @abc.abstractmethod
    def pattern(self) -> Pattern:
        ...

    @abc.abstractmethod
    def replace(self, m: "AsmMatch") -> Optional[Replacement]:
        ...

    def match(self, matcher: "AsmMatcher") -> Optional[Replacement]:
        m = matcher.try_match(self.pattern)
        if not m:
            return None
        return self.replace(m)


@dataclass
class TryMatchState:
    symbolic_registers: Dict[str, Register] = field(default_factory=dict)
    symbolic_labels: Dict[str, str] = field(default_factory=dict)
    symbolic_literals: Dict[str, int] = field(default_factory=dict)
    symbolic_instructions: Dict[str, Instruction] = field(default_factory=dict)

    T = TypeVar("T")

    def copy(self) -> "TryMatchState":
        return TryMatchState(
            self.symbolic_registers.copy(),
            self.symbolic_labels.copy(),
            self.symbolic_literals.copy(),
            self.symbolic_instructions.copy(),
        )

    def match_var(self, var_map: Dict[str, T], key: str, value: T) -> bool:
        if key in var_map:
            if var_map[key] != value:
                return False
        else:
            var_map[key] = value
        return True

    def match_reg(self, actual: Register, exp: Register) -> bool:
        if len(exp.register_name) <= 1:
            return self.match_var(self.symbolic_registers, exp.register_name, actual)
        else:
            return exp.register_name == actual.register_name

    def eval_math(self, e: Argument) -> int:
        if isinstance(e, AsmLiteral):
            return e.value
        if isinstance(e, BinOp):
            if e.op == "+":
                return self.eval_math(e.lhs) + self.eval_math(e.rhs)
            if e.op == "-":
                return self.eval_math(e.lhs) - self.eval_math(e.rhs)
            if e.op == "<<":
                return self.eval_math(e.lhs) << self.eval_math(e.rhs)
            assert False, f"bad binop in math pattern: {e}"
        elif isinstance(e, AsmGlobalSymbol):
            assert (
                e.symbol_name in self.symbolic_literals
            ), f"undefined variable in math pattern: {e.symbol_name}"
            return self.symbolic_literals[e.symbol_name]
        else:
            assert False, f"bad pattern part in math pattern: {e}"

    def match_arg(self, a: Argument, e: Argument) -> bool:
        if isinstance(e, AsmLiteral):
            return isinstance(a, AsmLiteral) and a.value == e.value
        if isinstance(e, Register):
            return isinstance(a, Register) and self.match_reg(a, e)
        if isinstance(e, AsmGlobalSymbol):
            if e.symbol_name.isupper():
                return isinstance(a, AsmLiteral) and self.match_var(
                    self.symbolic_literals, e.symbol_name, a.value
                )
            else:
                return isinstance(a, AsmGlobalSymbol) and a.symbol_name == e.symbol_name
        if isinstance(e, AsmAddressMode):
            return (
                isinstance(a, AsmAddressMode)
                and a.lhs == e.lhs
                and self.match_reg(a.rhs, e.rhs)
            )
        if isinstance(e, JumpTarget):
            return isinstance(a, JumpTarget) and self.match_var(
                self.symbolic_labels, e.target, a.target
            )
        if isinstance(e, BinOp):
            return isinstance(a, AsmLiteral) and a.value == self.eval_math(e)
        assert False, f"bad pattern part: {e}"

    def match_one(self, actual: BodyPart, exp: BodyPart) -> bool:
        if isinstance(exp, Label):
            return isinstance(actual, Label) and self.match_var(
                self.symbolic_labels, exp.name, actual.name
            )
        if not isinstance(actual, Instruction):
            return False
        ins = actual
        if exp.mnemonic.startswith("*"):
            self.symbolic_instructions[exp.mnemonic[1:]] = ins
        elif ins.mnemonic != exp.mnemonic:
            return False
        if exp.args:
            if len(ins.args) != len(exp.args):
                return False
            for (a, e) in zip(ins.args, exp.args):
                if not self.match_arg(a, e):
                    return False
        return True


@dataclass
class AsmMatcher:
    remaining: List[BodyPart]
    output: List[BodyPart] = field(default_factory=list)
    unique_ctr: int = 0

    def try_match(
        self, pattern: Pattern, allow_reorder: bool = False
    ) -> Optional[AsmMatch]:
        state = TryMatchState()

        pati = 0
        index = len(self.remaining) - 1
        consumed = []
        unrelated = []
        while pati < len(pattern):
            exp, optional = pattern[pati]
            if index < 0:
                if not optional:
                    return None
                break
            saved_state = state.copy()
            actual = self.remaining[index]
            if state.match_one(actual, exp):
                consumed.append(actual)
                index -= 1
                pati += 1
            else:
                state = saved_state
                if optional:
                    pati += 1
                elif allow_reorder and consumed and not isinstance(actual, Label):
                    unrelated.append(actual)
                    index -= 1
                else:
                    return None
        return AsmMatch(
            consumed,
            unrelated,
            state.symbolic_registers,
            state.symbolic_literals,
            state.symbolic_instructions,
        )

    def unique_reg(self) -> Register:
        self.unique_ctr += 1
        return Register(f"fakereg{self.unique_ctr}")

    def apply(self, repl: Replacement) -> None:
        for _ in range(repl.num_consumed):
            self.remaining.pop()
        for part in repl.new_body[::-1]:
            self.remaining.append(part)

    def skip(self) -> None:
        self.output.append(self.remaining.pop())


def simplify_patterns(
    body: List[BodyPart], patterns: List[AsmPattern]
) -> List[BodyPart]:
    """Detect and simplify asm standard patterns emitted by known compilers. This is
    especially useful for patterns that involve branches, which are hard to deal with
    in the translate phase."""
    matcher = AsmMatcher(body[::-1])
    while matcher.remaining:
        for pattern in patterns:
            m = pattern.match(matcher)
            if m:
                matcher.apply(m)
                break
        else:
            matcher.skip()

    return matcher.output
