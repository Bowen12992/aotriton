import hashlib
import json
import sys
from argparse import ArgumentParser
from multiprocessing import Process, Queue
from pathlib import Path
from typing import List

from triton.backends.compiler import GPUTarget

KNOWN_TARGETS = {
    None: None,
    "A100": GPUTarget("cuda", 80, 32),
    "RTX3090": GPUTarget("cuda", 86, 32),
}

desc = """
Triton ahead-of-time compiler:
"""


def do_compile(args):
    import triton
    from triton.backends.amd.driver import ty_to_cpp
    from triton.compiler.code_generator import kernel_suffix

    out_path = args.out_path

    # execute python sources and extract functions wrapped in JITFunction
    arg_path = Path(args.path)
    sys.path.insert(0, str(arg_path.parent))
    """
    spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, args.kernel_name)
    """
    if True:
        exec_string = f"import {arg_path.stem}"
        # print(exec_string)
        exec(exec_string, globals())  # importlib code path miss things
        # print(globals())
        # kernel = globals()[f"{arg_path.stem}.{args.kernel_name}"]
        mod = globals()[arg_path.stem]
        kernel = getattr(mod, args.kernel_name)

    grid = args.grid.split(",")
    assert len(grid) == 3

    # validate and parse signature
    signature = list(map(lambda s: s.strip(" "), args.signature.split(",")))

    def hash_signature(signature: List[str]):
        m = hashlib.sha256()
        m.update(" ".join(signature).encode())
        return m.hexdigest()[:8]

    meta_sig = f"warps{args.num_warps}xstages{args.num_stages}"
    hash_signature(signature + [meta_sig])

    def constexpr(s):
        try:
            ret = int(s)
            return ret
        except ValueError:
            pass
        try:
            ret = float(s)
            return ret
        except ValueError:
            pass
        if s == "True":
            return True
        if s == "False":
            return False
        return None

    hints = {i: constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    constants = {kernel.arg_names[i]: constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}
    signature = {
        kernel.arg_names[i]: s.split(":")[0]
        for i, s in enumerate(signature)
        if kernel.arg_names[i] not in constants
    }
    "x".join([str(v) for v in constants.values()])
    doc_string = [f"{k}={v}" for k, v in constants.items()]
    doc_string += [f"num_warps={args.num_warps}", f"num_stages={args.num_stages}"]

    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    attrs = triton.backends.compiler.AttrsDescriptor.from_hints(hints)
    for p, v in attrs.get_constants().items():
        constants.update({kernel.arg_names[p]: v})
    src = triton.compiler.ASTSource(
        fn=kernel, constants=constants, signature=signature, attrs=attrs
    )
    opts = {"num_warps": args.num_warps, "num_stages": args.num_stages}
    ccinfo = triton.compile(src, target=KNOWN_TARGETS[args.target], options=opts)
    # import pdb; pdb.set_trace()
    with open(out_path.with_suffix(".cubin"), "bw") as f:
        f.write(ccinfo.kernel)
    with open(out_path.with_suffix(".json"), "w") as f:
        di = ccinfo.metadata._asdict()
        del di["target"]  # Cannot be serialized to Json
        di["compile_status"] = "Complete"
        json.dump(di, f, indent=2)
    return out_path


def ipc_compile(ipc_in, ipc_out):
    args = ipc_in.get()
    try:
        do_compile(args)
        ipc_out.put("Complete")
    except Exception as e:
        if args.verbose:
            print(e)
        ipc_out.put("Exception")


def main_fn(args):
    # command-line arguments
    if args.timeout <= 0:
        do_compile(args)
        return
    ipc_to_worker = Queue()
    ipc_worker_out = Queue()
    ipc_to_worker.cancel_join_thread()
    ipc_worker_out.cancel_join_thread()
    worker = Process(target=ipc_compile, args=(ipc_to_worker, ipc_worker_out))
    worker.start()
    ipc_to_worker.put(args)
    worker.join(args.timeout * 60.0)
    if worker.exitcode == 0:
        status = ipc_worker_out.get()
    elif worker.exitcode is None:
        worker.kill()
        status = "Timeout"
    else:
        status = "ExitWithError"
    if status == "Timeout":
        print(
            f"Compiling {args.path=} {args.kernel_name} to {args.out_path=} timed out with {args.timeout} minutes",
            file=sys.stderr,
        )
    ipc_to_worker.close()
    ipc_worker_out.close()
    if args.verbose and status == "ExitWithError":
        print(
            f"Compiling {args.path=} {args.kernel_name} to {args.out_path=} result with status {status} exitcode {worker.exitcode}"
        )
    # Write an empty file to avoid errors
    if status != "Complete":
        with open(args.out_path.with_suffix(".cubin"), "bw") as f:
            pass
        with open(args.out_path.with_suffix(".json"), "w") as f:
            d = {"compile_status": status}
            json.dump(d, f, indent=2)
