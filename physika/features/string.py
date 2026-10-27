from physika.elf import ELF
from typing import Callable, Optional, Tuple
from physika.utils.types import Substitution, Type, T_STRING


def make_parser_rules():
    """
    PLY grammar functions for Physika string dtype.
    """

    def p_factor_string(p):
        """factor : STRING"""
        # String literal in an expression
        # Example:
        #   "physika"
        # Parameters:
        # p[1] - string literal value
        # Returns:
        #   ("string", value)
        p[0] = ("string", p[1])

    def p_func_factor_string(p):
        """func_factor : STRING"""
        # String literal in a function expression
        # Example:
        #   "physika"
        # Parameters:
        # p[1] - string literal value
        # Returns:
        #   ("string", value)
        p[0] = ("string", p[1])

    return [p_factor_string, p_func_factor_string]


class StringFeature(ELF):
    """
    Physika string dtype implemented as ELF subclass.

    ``StringFeature`` oinjects rules via ``REGISTRY`` at parser,
    type checker.

    **Parser rules**
    Two PLY grammer functions (see ``make_parser_rules``) handles
    string declarations for both top level and function level

    **Type rules**
    Registers ``expr_string`` so the type checker is aware of string.

    Physika syntax example (see ``examples/example_strings.phyk``)::

        name: string = "Physika"

    Examples
    --------
    >>> from physika.lexer import lexer
    >>> from physika.parser import parser, symbol_table
    >>> from physika.utils.ast_utils import build_unified_ast
    >>> from physika.codegen import from_ast_to_torch
    >>> def run_phyk(src):
    ...     symbol_table.clear()
    ...     lexer.lexer.lineno = 1
    ...     ast = build_unified_ast(parser.parse(src, lexer=lexer),
    ...                                         symbol_table)
    ...     exec(from_ast_to_torch(ast, print_code=False), {})

    >>> # Physika string example
    >>> src = '''
    ... name: string = "Physika"
    ... '''
    >>> # Execute code and verify outputs
    >>> run_phyk(src)
    Physika ∈ string
    """
    name = "string"

    def parser_rules(self) -> list:
        """
        Override ``parser_rules`` handler for new grammar rules.

        Returns
        -------
        list
            List of PLY grammar functions to be injected into
            ``physika.parser``.

        Examples
        --------
        >>> from physika.features import StringFeature
        >>> rules = StringFeature().parser_rules()
        >>> len(rules)
        2
        >>> rules[0].__name__
        'p_factor_string'
        """
        return make_parser_rules()

    def type_rules(self) -> dict:
        """
        Registers one type-checking handler that return T_STRING type.

        Returns
        -------
        dict
            Dispatch table mapping ``"string"`` AST tag to its
            type inference handler.

        Examples
        --------
        >>> from physika.features import StringFeature
        >>> rules = StringFeature().type_rules()
        >>> sorted(rules.keys())
        ['string']
        """

        def expr_string(
                node: tuple, env: dict, s: Substitution, func_env: dict,
                class_env: dict, add_error: Callable,
                infer_expr: Callable) -> Tuple[Optional[Type], Substitution]:
            """
            The type of string literal is always ``String``.

            Parameters
            ----------
            node : Tuple
                AST node of the form ``("string", value)`` where *value* is an
                ``String``.
            ctx : ExprContext
                Current inference context.

            Returns
            -------
            tuple[Type, Substitution]
                Always ``(T_STRING, ctx.s)``.

            Examples
            --------
            >>> from physika.utils.infer_expr import ExprContext, expr_string
            >>> from physika.utils.types import Substitution, T_STRING
            >>> ctx = ExprContext({}, Substitution(), {}, {}, [].append)
            >>> t, _= expr_string(("string", "physika"), ctx)
            >>> t
            string
            """
            return T_STRING, s

        return {"string": expr_string}
