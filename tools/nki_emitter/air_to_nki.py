#!/usr/bin/env python3
# tools/nki_emitter/air_to_nki.py
#
# Reads an AIR-dialect MLIR file, runs the standard lowering pipeline via
# air-opt (air-to-std, lower-affine, canonicalize, cse), then walks the
# resulting func/scf/memref/arith IR and emits equivalent NKI Python.
#
# Usage:
#   PYTHONPATH=my_install/python python3 tools/nki_emitter/air_to_nki.py \
#       test/gpu/simple_test/simple_test.mlir [-o out.py]

import argparse
import os
import shutil
import subprocess
import sys

import air.ir as ir

# ---------------------------------------------------------------------------
# Lowering via air-opt subprocess
# ---------------------------------------------------------------------------

LOWER_FLAGS = ["--air-to-std", "--lower-affine", "--canonicalize", "--cse"]


def _find_air_opt(hint: str | None) -> str:
    if hint:
        return hint
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(here, "..", "..", "build", "bin", "air-opt"))
    if os.path.isfile(candidate):
        return candidate
    found = shutil.which("air-opt")
    if found:
        return found
    raise FileNotFoundError("Cannot find air-opt. Use --air-opt <path> to specify it.")


def lower_mlir(src: str, air_opt: str) -> str:
    result = subprocess.run(
        [air_opt] + LOWER_FLAGS + ["-"],
        input=src, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("air-opt failed:\n", result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout


# ---------------------------------------------------------------------------
# MLIR type helpers
# ---------------------------------------------------------------------------

def _type_to_nki(t: ir.Type) -> str:
    mapping = {
        "f32": "nl.float32", "f16": "nl.float16", "bf16": "nl.bfloat16",
        "i32": "nl.int32",   "i16": "nl.int16",   "i8":   "nl.int8",
        "i1":  "nl.bool_",
    }
    return mapping.get(str(t), str(t))


def _memref_info(t: ir.Type):
    """Return (shape_tuple, dtype_str) for a MemRefType."""
    mrt = ir.MemRefType(t)  # raises if not a memref
    return tuple(mrt.shape), _type_to_nki(mrt.element_type)


def _is_memref(t: ir.Type) -> bool:
    try:
        ir.MemRefType(t)
        return True
    except Exception:
        return False


def _attr_get(op: ir.Operation, key: str):
    """Safe attribute lookup — OpAttributeMap has no .get()."""
    return op.attributes[key] if key in op.attributes else None


# ---------------------------------------------------------------------------
# Visitor-pattern NKI emitter
# ---------------------------------------------------------------------------

class NKIEmitter:
    """
    Walks a lowered MLIR module and emits NKI Python.

    Dispatch is based on the visitor pattern: for each MLIR op whose
    operation name is ``dialect.op_name``, the method
    ``visit_dialect_op_name`` is called (dots replaced by underscores).
    If no specific visitor exists, ``visit_default`` is called.
    """

    # arith op → numpy ufunc for nisa.tensor_tensor
    _NUMPY_OPS = {
        "arith.addf": "np.add",      "arith.addi": "np.add",
        "arith.subf": "np.subtract", "arith.subi": "np.subtract",
        "arith.mulf": "np.multiply", "arith.muli": "np.multiply",
        "arith.divf": "np.divide",
    }

    def __init__(self):
        self._lines: list[str] = []
        self._indent = 0
        self._vmap: dict[int, str] = {}   # ir.Value id → Python name/literal
        self._counter = 0

    # -----------------------------------------------------------------------
    # Core visitor dispatch
    # -----------------------------------------------------------------------

    def visit(self, op: ir.Operation):
        """Dispatch to visit_<dialect>_<opname>, falling back to visit_default."""
        op_name = op.operation.name          # e.g. "scf.for", "func.func"
        method_name = "visit_" + op_name.replace(".", "_")
        visitor = getattr(self, method_name, self.visit_default)
        visitor(op)

    def visit_default(self, op: ir.Operation):
        self._emit(f"# [unhandled op: {op.operation.name}]")

    # -----------------------------------------------------------------------
    # Output helpers
    # -----------------------------------------------------------------------

    def _emit(self, line: str):
        self._lines.append("    " * self._indent + line)

    def _blank(self):
        self._lines.append("")

    def _push(self): self._indent += 1
    def _pop(self):  self._indent -= 1

    def _fresh(self, hint: str = "v") -> str:
        self._counter += 1
        return f"{hint}_{self._counter}"

    def _bind(self, val: ir.Value, name: str):
        self._vmap[id(val)] = name

    def _name(self, val: ir.Value) -> str:
        key = id(val)
        if key not in self._vmap:
            self._vmap[key] = self._fresh()
        return self._vmap[key]

    # -----------------------------------------------------------------------
    # Module entry-point
    # -----------------------------------------------------------------------

    def emit_module(self, module: ir.Module) -> str:
        self._emit("# Auto-generated NKI kernel")
        self._emit("# Generated by tools/nki_emitter/air_to_nki.py")
        self._blank()
        self._emit("import numpy as np")
        self._emit("import neuronxcc.nki as nki")
        self._emit("import neuronxcc.nki.language as nl")
        self._emit("import neuronxcc.nki.isa as nisa")
        self._blank()
        for op in module.body.operations:
            self.visit(op)
        return "\n".join(self._lines)

    # -----------------------------------------------------------------------
    # Visitors
    # -----------------------------------------------------------------------

    def visit_func_func(self, op: ir.Operation):
        func_name = ir.StringAttr(op.attributes["sym_name"]).value
        entry: ir.Block = op.regions[0].blocks[0]

        arg_parts = []
        for i, arg in enumerate(entry.arguments):
            name = f"arg{i}"
            self._bind(arg, name)
            if _is_memref(arg.type):
                shape, dtype = _memref_info(arg.type)
                arg_parts.append(f"{name}: nl.ndarray[{shape}, {dtype}]")
            else:
                arg_parts.append(name)

        self._emit("@nki.jit")
        self._emit(f"def {func_name}({', '.join(arg_parts)}):")
        self._push()
        for inner in entry.operations:
            self.visit(inner)
        self._pop()
        self._blank()

    def visit_func_return(self, op: ir.Operation):
        self._emit("return")

    def visit_arith_constant(self, op: ir.Operation):
        attr = _attr_get(op, "value")
        try:
            val_str = str(ir.IntegerAttr(attr).value)
        except Exception:
            try:
                val_str = repr(ir.FloatAttr(attr).value)
            except Exception:
                val_str = str(attr)
        # Inline constants as Python literals — no statement emitted
        self._bind(op.results[0], val_str)

    def visit_arith_index_cast(self, op: ir.Operation):
        # Transparent cast: alias result to operand
        self._bind(op.results[0], self._name(op.operands[0]))

    visit_arith_index_castui = visit_arith_index_cast

    def visit_scf_for(self, op: ir.Operation):
        lb   = self._name(op.operands[0])
        ub   = self._name(op.operands[1])
        step = self._name(op.operands[2])

        body: ir.Block = op.regions[0].blocks[0]
        iv = body.arguments[0]

        # Fold trivial 1-trip loops (artifacts of air.launch 1x1 grids)
        try:
            if int(lb) == 0 and int(ub) == 1 and int(step) == 1:
                self._bind(iv, "0")
                for inner in body.operations:
                    self.visit(inner)
                return
        except ValueError:
            pass

        iv_name = self._fresh("i")
        self._bind(iv, iv_name)
        self._emit(f"for {iv_name} in nl.affine_range({lb}, {ub}, {step}):")
        self._push()
        for inner in body.operations:
            self.visit(inner)
        self._pop()

    def visit_scf_yield(self, op: ir.Operation):
        pass  # No statement needed for a plain yield with no loop-carried values

    def visit_memref_load(self, op: ir.Operation):
        mem = self._name(op.operands[0])
        idx = ", ".join(self._name(op.operands[i]) for i in range(1, len(op.operands)))
        res = self._fresh("val")
        self._bind(op.results[0], res)
        self._emit(f"{res} = nl.load({mem}[{idx}])")

    def visit_memref_store(self, op: ir.Operation):
        val = self._name(op.operands[0])
        mem = self._name(op.operands[1])
        idx = ", ".join(self._name(op.operands[i]) for i in range(2, len(op.operands)))
        self._emit(f"nl.store({mem}[{idx}], value={val})")

    def _visit_arith_binop(self, op: ir.Operation):
        np_op = self._NUMPY_OPS[op.operation.name]
        lhs = self._name(op.operands[0])
        rhs = self._name(op.operands[1])
        res = self._fresh("res")
        self._bind(op.results[0], res)
        self._emit(f"{res} = nisa.tensor_tensor({lhs}, {rhs}, op={np_op})")

    # Wire all binary arith ops to the shared helper via the visitor protocol
    visit_arith_addf = _visit_arith_binop
    visit_arith_subf = _visit_arith_binop
    visit_arith_mulf = _visit_arith_binop
    visit_arith_divf = _visit_arith_binop
    visit_arith_addi = _visit_arith_binop
    visit_arith_subi = _visit_arith_binop
    visit_arith_muli = _visit_arith_binop


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Lower AIR-dialect MLIR to NKI Python.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input .mlir file (AIR dialect)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output .py file (default: stdout)")
    parser.add_argument("--air-opt", dest="air_opt", default=None,
                        help="Path to air-opt binary (auto-detected if not set)")
    parser.add_argument("--no-lower", action="store_true",
                        help="Skip air-opt lowering: input already in func/scf/memref/arith")
    args = parser.parse_args()

    with open(args.input) as f:
        src = f.read()

    if not args.no_lower:
        air_opt = _find_air_opt(args.air_opt)
        src = lower_mlir(src, air_opt)

    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(src)
        nki_src = NKIEmitter().emit_module(module)

    if args.output:
        with open(args.output, "w") as f:
            f.write(nki_src)
        print(f"Wrote NKI Python to: {args.output}", file=sys.stderr)
    else:
        print(nki_src)


if __name__ == "__main__":
    main()
