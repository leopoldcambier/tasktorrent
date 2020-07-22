ml intel/python
grep '\[0\]>>>>' -R *.out >ttor_gemm_2d_raw.dat
python3 time_to_eff.py
