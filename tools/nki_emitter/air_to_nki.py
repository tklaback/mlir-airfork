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

def _type_to_nki(t: ir.Type) -> str:
    mapping = {
        "f32": "nl.float32", "f16": "nl.float16", "bf16": "nl.bfloat16",
        "i32": "nl.int32",   "i16": "nl.int16",   "i8":   "nl.int8",
        "i1":  "nl.bool_",
    }
    return mapping.get(str(t), str(t))


def _memref_info(t: ir.Type):
    mrt = ir.MemRefType(t)
    return tuple(mrt.shape), _type_to_nki(mrt.element_type)

def _is_memref(t: ir.Type) -> bool:
    try:
        ir.MemRefType(t)
        return True
    except Exception:
        return False

def _attr_get(op: ir.Operation, key: str):
    return op.attributes[key] if key in op.attributes else None

class NKIEmitter:

    _NUMPY_OPS = {
        "arith.addf": "np.add",      "arith.addi": "np.add",
        "arith.subf": "np.subtract", "arith.subi": "np.subtract",
        "arith.mulf": "np.multiply", "arith.muli": "np.multiply",
        "arith.divf": "np.divide",
    }

    def __init__(self):
        self._lines: list[str] = []
        self._indent = 0
        self._vmap: dict[int, str] = {}
        self._counter = 0

    def visit(self, op: ir.Operation):
        op_name = op.operation.name
        method_name = "visit_" + op_name.replace(".", "_")
        visitor = getattr(self, method_name, self.visit_default)
        visitor(op)

    def visit_default(self, op: ir.Operation):
        self._emit(f"# [unhandled op: {op.operation.name}]")

    def _emit(self, line: str):
        self._lines.append("    " * self._indent + line)

    def _blank(self):
        self._lines.append("")

    def _push(self): self._indent += 1
    def _pop(self):  self._indent -= 1

    def _bind(self, val: ir.Value, name: str):
        self._vmap[id(val)] = name

    def emit_module(self, module: ir.Module) -> str:
        for op in module.body.operations:
            self.visit(op)
        return "\n".join(self._lines)

    def visit_func_func(self, op: ir.Operation):
        func_name = ir.StringAttr(op.attributes["sym_name"]).value
        entry: ir.Block = op.regions[0].blocks[0]
        args = list(entry.arguments)

        input_names: list[str] = []
        idx = 0
        for arg in args:
            name = f"arg{idx}"
            idx += 1
            self._bind(arg, name)
            input_names.append(name)

        self._emit("@nki.jit")
        self._emit(f"def {func_name}({', '.join(input_names)}):")
        self._push()

        for inner in entry.operations:
            self.visit(inner)
        self._pop()
        self._blank()

    def visit_func_return(self, op: ir.Operation):
        pass

    def visit_arith_constant(self, op: ir.Operation):
        ssa_name = op.results[0].get_name()[1:]
        attr = _attr_get(op, "value")
        val_str = ir.IntegerAttr(attr).value
        self._bind(ssa_name, val_str)

    def visit_arith_index_cast(self, op: ir.Operation):
        pass

    def visit_scf_for(self, op: ir.Operation):
        pass

    def visit_scf_yield(self, op: ir.Operation):
        pass

    def visit_memref_load(self, op: ir.Operation):
        pass

    def visit_memref_store(self, op: ir.Operation):
        pass

    def _visit_arith_binop(self, op: ir.Operation):
        pass

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
