from physika.features.string import StringFeature
from tests.conftest import exec_phyk
from tests.test_utils.test_infer_expr import make_ctx
from physika.utils.types import T_STRING, Substitution, T_NAT
from physika.utils.infer_expr import infer_expr
import pytest

type_rules = StringFeature().type_rules()
expr_string = type_rules["string"]


@pytest.fixture(scope="module")
def strings_ns():
    """
    Execute example_strings.phyk, build unified AST, execute; return
    namespace.
    """
    return exec_phyk("example_strings")


class TestStringFeature:
    """
    Checks parser is emitting the correct AST nodes,
    that the forward rules have the correct keys and type
    rules catches proper errors.
    """

    def test_elf_name(self):
        """
        ELF is registered under the name 'string'.
        """
        assert StringFeature.name == "string"

    def test_parser_rules(self):
        """
        parser_rules should return two handlers.
        """
        rules = StringFeature().parser_rules()
        assert len(rules) == 2
        names = [r.__name__ for r in rules]
        assert "p_factor_string" in names
        assert "p_func_factor_string" in names

    def test_forward_rules_keys(self):
        """
        type rules should return one handler.
        """
        rules = StringFeature().type_rules()
        assert len(rules) == 1
        assert set(rules.keys()) == {"string"}
        assert all(callable(v) for v in rules.values())


class TestExprString:
    """
    Tests for ``expr_string``
    """

    def test_string(self):
        """A string literal must infer to String."""
        ctx = make_ctx()
        t, _ = expr_string(("string", "physika"), ctx.env, ctx.s, ctx.func_env,
                           ctx.class_env, ctx.add_error, infer_expr)
        assert t == T_STRING

    def test_empty_string(self):
        """An empty string must infer to String."""
        ctx = make_ctx()
        t, _ = expr_string(("string", ""), ctx.env, ctx.s, ctx.func_env,
                           ctx.class_env, ctx.add_error, infer_expr)
        assert t == T_STRING

    def test_string_substitution_unchanged(self):
        """The substitution must remain unchanged."""
        existing = Substitution({"α0": T_NAT})
        ctx = make_ctx(s=existing)
        _, s_out = expr_string(("string", "hello"), ctx.env, ctx.s,
                               ctx.func_env, ctx.class_env, ctx.add_error,
                               infer_expr)
        assert s_out == existing

    def test_string_with_spaces(self):
        """A string containing spaces must infer to String."""
        ctx = make_ctx()
        t, _ = expr_string(("string", "x0 = a + b"), ctx.env, ctx.s,
                           ctx.func_env, ctx.class_env, ctx.add_error,
                           infer_expr)
        assert t == T_STRING


class TestStrings:
    """
    Test suite for ``examples/example_strings.phyk`` file.
    """

    def test_string_dtype(self, strings_ns):
        """
        Tests for string data type,
        * declarations
        * concatenation
        * indexing
        * slicing
        """

        # declarations
        empty_string = strings_ns["empty_string"]
        assert empty_string == ""

        name = strings_ns["name"]
        assert name == "Physika"

        msg = strings_ns["msg"]
        assert msg == "Hello, World!"

        # concatenation
        full_name = strings_ns["full_name"]
        assert full_name == "Physika Language"

        # indexing
        first_character = strings_ns["first_character"]
        assert first_character == "P"

        third_character = strings_ns["third_character"]
        assert third_character == "y"

        # slicing
        first_part = strings_ns["first_part"]
        assert first_part == "Physika"

        last_part = strings_ns["last_part"]
        assert last_part == "Language"

        full_slice = strings_ns["full_slice"]
        assert full_slice == "Physika Language"

        neg_slize = strings_ns["neg_slice"]
        assert neg_slize == "e"
