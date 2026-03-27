



@nki.jit
def single_put_get(a0, a1):
  current_rank = nl.program_id()
  i = current_rank // 2
  j = current_rank % 2
  row_offset = i * 8
  col_offset = j * 8
  tile = a0[row_offset:row_offset+8, col_offset:col_offset+16]
  
  scratch = nl.load(tile)
  scratch_copy = nl.ndarray(scratch.shape, dtype=scratch.dtype, buffer=nl.sbuf)
  for ii in range(8):
      for jj in range(16):
          scratch_copy[ii, jj] = scratch[ii, jj]
  nl.store(a1[row_offset:row_offset+8, col_offset:col_offset+16], scratch_copy)



a0 = np.ndarray(32, 16)
a1 = np.ndarray(32, 16)
single_put_get[nl.par_dims(4)](a0, a1)

  # allocate memory in sbuf for scratch and scratch_copy
  # recv from channel into scratch (not async)
  # iterate over 8x16 chunk, load from scratch and store in scratch copy
  # send back through channel 1 (not async)
  # dependency graph: channel1get -> channel1put -> channel0get -> channel0put

  # This example is not a cross-herd example and so when we represent channels, we can do so with just simple loads and stores to and from sbuf. Otherwise, sendrecv would probably be better, but is overkill for this example.




# Should I have a producer kernel and a consumer model? no