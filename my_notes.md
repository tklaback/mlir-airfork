## Notes

### air-to-nki

1. ./build/bin/air-opt \
  --air-to-std \
  --lower-affine \
  --canonicalize \
  --cse \
  test/gpu/simple_test/simple_test.mlir -o lowered.mlir