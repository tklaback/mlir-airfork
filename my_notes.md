## Notes

### air-to-nki

1. ./build/bin/air-opt \
  --air-to-std \
  --lower-affine \
  --canonicalize \
  --cse \
  test/gpu/simple_test/simple_test.mlir -o lowered.mlir

2. PYTHONPATH=/home/ty/mlir-air/my_install/python python3 /home/ty/mlir-air/tools/nki_emitter/air_to_nki.py test/gpu/simple_test/simple_test.mlir

3. PYTHONPATH=/home/ty/mlir-air/my_install/python python3 /home/ty/mlir-air/tools/nki_emitter/air_to_nki.py /home/ty/mlir-air/tools/air-translate/test.lit