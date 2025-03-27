# MiniTorch

Almost all commits was launched after use `black` to format.

All the problems I encountered are showed in commit-info and the code comments, please enjoy!

## diagnostics of `parallel_check.py`

```shell
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
e:\000download\github\minitorch\minitorch\fast_ops.py (159)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, e:\000download\github\minitorch\minitorch\fast_ops.py (159)
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        in_storage: Storage,                                                 |
        in_shape: Shape,                                                     |
        in_strides: Strides,                                                 |
    ) -> None:                                                               |
        # TODO: Implement for Task 3.1.                                      |
        # array -> .all()                                                    |
        if (                                                                 |
            len(out_shape) == len(in_shape)                                  |
            and (out_strides == in_strides).all()----------------------------| #0
            and (out_shape == in_shape).all()--------------------------------| #1
        ):                                                                   |
            for i in prange(len(out)):---------------------------------------| #2
                out[i] = fn(in_storage[i])                                   |
        else:                                                                |
            for out_ordinal in prange(len(out)):-----------------------------| #3
                out_index, in_index = np.empty(                              |
                    len(out_shape), dtype=np.int32                           |
                ), np.empty(len(in_shape), dtype=np.int32)                   |
                to_index(out_ordinal, out_shape, out_index)                  |
                broadcast_index(out_index, out_shape, in_shape, in_index)    |
                in_ordinal = index_to_position(in_index, in_strides)         |
                out[out_ordinal] = fn(in_storage[in_ordinal])                |
        return                                                               |
        raise NotImplementedError("Need to implement for Task 3.1")          |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
e:\000download\github\minitorch\minitorch\fast_ops.py (178) is hoisted out of
the parallel loop labelled #3 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: out_index, in_index = np.empty(
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
e:\000download\github\minitorch\minitorch\fast_ops.py (180) is hoisted out of
the parallel loop labelled #3 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: ), np.empty(len(in_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
e:\000download\github\minitorch\minitorch\fast_ops.py (213)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, e:\000download\github\minitorch\minitorch\fast_ops.py (213)
---------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                          |
        out: Storage,                                                                  |
        out_shape: Shape,                                                              |
        out_strides: Strides,                                                          |
        a_storage: Storage,                                                            |
        a_shape: Shape,                                                                |
        a_strides: Strides,                                                            |
        b_storage: Storage,                                                            |
        b_shape: Shape,                                                                |
        b_strides: Strides,                                                            |
    ) -> None:                                                                         |
        # TODO: Implement for Task 3.1.                                                |
        # [[1]] == [1]                                                                 |
        f1, f2 = (                                                                     |
            len(out_shape) == len(a_shape)                                             |
            and (out_strides == a_strides).all()---------------------------------------| #4
            and (out_shape == a_shape).all(),------------------------------------------| #5
            len(out_shape) == len(b_shape)                                             |
            and (out_strides == b_strides).all()---------------------------------------| #6
            and (out_shape == b_shape).all(),------------------------------------------| #7
        )                                                                              |
        if f1 and f2:                                                                  |
            for i in prange(len(out)):-------------------------------------------------| #8
                out[i] = fn(a_storage[i], b_storage[i])                                |
        else:                                                                          |
            for out_ordinal in prange(len(out)):---------------------------------------| #9
                out_index = np.empty(len(out_shape), dtype=np.int32)                   |
                a_ordinal, b_ordinal = out_ordinal, out_ordinal                        |
                if not f1 or not f2:                                                   |
                    to_index(out_ordinal, out_shape, out_index)                        |
                if not f1:                                                             |
                    a_index = np.empty(len(a_shape), dtype=np.int32)                   |
                    broadcast_index(out_index, out_shape, a_shape, a_index)            |
                    a_ordinal = index_to_position(a_index, a_strides)                  |
                if not f2:                                                             |
                    b_index = np.empty(len(b_shape), dtype=np.int32)                   |
                    broadcast_index(out_index, out_shape, b_shape, b_index)            |
                    b_ordinal = index_to_position(b_index, b_strides)                  |
                # NumbaTypeError: Unsupported array index type float64 in [float64]    |
                out[int(out_ordinal)] = fn(                                            |
                    a_storage[int(a_ordinal)], b_storage[int(b_ordinal)]               |
                )                                                                      |
        return                                                                         |
        raise NotImplementedError("Need to implement for Task 3.1")                    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #8, #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
e:\000download\github\minitorch\minitorch\fast_ops.py (248) is hoisted out of
the parallel loop labelled #9 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: b_index = np.empty(len(b_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
e:\000download\github\minitorch\minitorch\fast_ops.py (244) is hoisted out of
the parallel loop labelled #9 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: a_index = np.empty(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
e:\000download\github\minitorch\minitorch\fast_ops.py (239) is hoisted out of
the parallel loop labelled #9 (it will be performed before the loop is executed
and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
e:\000download\github\minitorch\minitorch\fast_ops.py (280)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, e:\000download\github\minitorch\minitorch\fast_ops.py (280)
--------------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                            |
        out: Storage,                                                                       |
        out_shape: Shape,                                                                   |
        out_strides: Strides,                                                               |
        a_storage: Storage,                                                                 |
        a_shape: Shape,                                                                     |
        a_strides: Strides,                                                                 |
        reduce_dim: int,                                                                    |
    ) -> None:                                                                              |
        # TODO: Implement for Task 3.1.                                                     |
        # hypothesis.errors.FailedHealthCheck, I add `a_shape[reduce_dim] == 1` to solve    |
        if a_shape[reduce_dim] == 1:                                                        |
            for i in prange(len(a_storage)):------------------------------------------------| #10
                out[i] = a_storage[i]                                                       |
        else:                                                                               |
            # for a_ordinal in prange(len(a_storage)):                                      |
            #     a_index = np.empty(len(a_shape), dtype=np.int32)                          |
            #     out_index = np.empty(len(a_shape), dtype=np.int32)                        |
            #     to_index(a_ordinal, a_shape, a_index)                                     |
            #     broadcast_index(a_index, a_shape, out_shape, out_index)                   |
            #     out_ordinal = index_to_position(out_index, out_strides)                   |
            #     out[out_ordinal] = fn(out[out_ordinal], a_storage[a_ordinal])             |
            for out_ordinal in prange(len(out)):--------------------------------------------| #11
                out_index = np.empty(len(out_shape), dtype=np.int32)                        |
                to_index(out_ordinal, out_shape, out_index)                                 |
                total = out[out_ordinal]                                                    |
                for k in range(a_shape[reduce_dim]):                                        |
                    a_index = out_index                                                     |
                    a_index[reduce_dim] = k                                                 |
                    a_ordinal = index_to_position(a_index, a_strides)                       |
                    total = fn(total, a_storage[a_ordinal])                                 |
                out[out_ordinal] = total                                                    |
        return                                                                              |
        raise NotImplementedError("Need to implement for Task 3.1")                         |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
e:\000download\github\minitorch\minitorch\fast_ops.py (303) is hoisted out of
the parallel loop labelled #11 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
e:\000download\github\minitorch\minitorch\fast_ops.py (318)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, e:\000download\github\minitorch\minitorch\fast_ops.py (318)
---------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                               |
    out: Storage,                                                          |
    out_shape: Shape,                                                      |
    out_strides: Strides,                                                  |
    a_storage: Storage,                                                    |
    a_shape: Shape,                                                        |
    a_strides: Strides,                                                    |
    b_storage: Storage,                                                    |
    b_shape: Shape,                                                        |
    b_strides: Strides,                                                    |
) -> None:                                                                 |
    """                                                                    |
    NUMBA tensor matrix multiply function.                                 |
                                                                           |
    Should work for any tensor shapes that broadcast as long as            |
                                                                           |
    ```                                                                    |
    assert a_shape[-1] == b_shape[-2]                                      |
    ```                                                                    |
                                                                           |
    Optimizations:                                                         |
                                                                           |
    * Outer loop in parallel                                               |
    * No index buffers or function calls                                   |
    * Inner loop should have no global writes, 1 multiply.                 |
                                                                           |
                                                                           |
    Args:                                                                  |
        out (Storage): storage for `out` tensor                            |
        out_shape (Shape): shape for `out` tensor                          |
        out_strides (Strides): strides for `out` tensor                    |
        a_storage (Storage): storage for `a` tensor                        |
        a_shape (Shape): shape for `a` tensor                              |
        a_strides (Strides): strides for `a` tensor                        |
        b_storage (Storage): storage for `b` tensor                        |
        b_shape (Shape): shape for `b` tensor                              |
        b_strides (Strides): strides for `b` tensor                        |
                                                                           |
    Returns:                                                               |
        None : Fills in `out`                                              |
    """                                                                    |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                 |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                 |
                                                                           |
    # TODO: Implement for Task 3.2.                                        |
    batch_size = out_shape[0]                                              |
    M = out_shape[1]                                                       |
    N = out_shape[2]                                                       |
    K = a_shape[-1]                                                        |
    for i in prange(batch_size):-------------------------------------------| #12
        a_batch_offset = i * a_batch_stride                                |
        b_batch_offset = i * b_batch_stride                                |
        for j in range(M):                                                 |
            a_row_offset = a_batch_offset + j * a_strides[-2]              |
            for k in range(N):                                             |
                b_col_offset = b_batch_offset + k * b_strides[-1]          |
                tot = 0.0                                                  |
                for u in range(K):                                         |
                    a_ordinal = a_row_offset + u * a_strides[-1]           |
                    b_ordinal = b_col_offset + u * b_strides[-2]           |
                    tot += a_storage[a_ordinal] * b_storage[b_ordinal]     |
                out_ordinal = index_to_position([i, j, k], out_strides)    |
                out[int(out_ordinal)] = tot                                |
    return                                                                 |
    raise NotImplementedError("Need to implement for Task 3.2")            |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #12).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
