"""Tutorial-local precision lowering for unchanged Physika-generated code.

The baseline compiler emits float32 casts and relies on Torch's ambient
dtype for some constructors. This pass binds both to one tutorial execution
without changing the compiler or monkey-patching its runtime.
"""
import ast

import torch


def as_typed_tensor(value, *, real_dtype, **kwargs):
    """Preserve booleans/complex values and explicitly construct real values."""
    sample = torch.as_tensor(value)
    if sample.is_complex():
        dtype = (torch.complex128 if real_dtype == torch.float64 else
                 torch.complex64)
    elif sample.dtype == torch.bool:
        dtype = torch.bool
    else:
        dtype = real_dtype
    return torch.as_tensor(value, dtype=dtype, **kwargs)


def lower_precision(code, precision):
    """Return generated Python with dtype-explicit casts and constructors.

    The returned code needs ``as_typed_tensor`` in its execution namespace.
    Functions keep the selected dtype after the ambient default changes.
    """
    if precision not in ('float32', 'float64'):
        raise ValueError('precision must be float32 or float64')

    def dtype_node(name=precision):
        return ast.Attribute(value=ast.Name(id='torch', ctx=ast.Load()),
                             attr=name, ctx=ast.Load())

    class Precision(ast.NodeTransformer):
        def visit_Attribute(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == 'torch':
                if node.attr == 'float32':
                    node.attr = precision
                elif node.attr == 'complex64' and precision == 'float64':
                    node.attr = 'complex128'
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if not isinstance(node.func, ast.Attribute):
                return node
            f = node.func
            if f.attr == 'float' and not node.args and not node.keywords:
                f.attr = 'to'
                node.keywords.append(ast.keyword(arg='dtype',
                                                 value=dtype_node()))
            is_torch = isinstance(f.value, ast.Name) and f.value.id == 'torch'
            explicit = any(k.arg == 'dtype' for k in node.keywords)
            if is_torch and not explicit:
                if f.attr == 'as_tensor':
                    node.func = ast.Name(id='as_typed_tensor', ctx=ast.Load())
                    node.keywords.append(ast.keyword(arg='real_dtype',
                                                     value=dtype_node()))
                elif f.attr in ('tensor', 'zeros', 'ones', 'arange', 'linspace'):
                    values = list(ast.walk(node))
                    complex_value = any(isinstance(v, ast.Constant) and
                                        isinstance(v.value, complex)
                                        for v in values)
                    boolean_value = (bool(node.args) and
                                     isinstance(node.args[0], ast.Constant) and
                                     isinstance(node.args[0].value, bool))
                    dtype = ('complex128' if precision == 'float64' else
                             'complex64') if complex_value else (
                                 'bool' if boolean_value else precision)
                    node.keywords.append(ast.keyword(arg='dtype',
                                                     value=dtype_node(dtype)))
            return node

    tree = Precision().visit(ast.parse(code))
    return ast.unparse(ast.fix_missing_locations(tree))
