from src.theory.insufficiency import reproduce_two_moment_insufficiency

def test_two_moment_prop():
 r=reproduce_two_moment_insufficiency()
 assert abs(r["mu1_mean"]-0.6)<1e-8
 assert abs(r["mu2_mean"]-0.6)<1e-8
 assert abs(r["mu1_var"]-0.04)<1e-8
 assert abs(r["mu2_var"]-0.04)<1e-8
 assert abs(r["R3_mu1"]-0.376)<1e-8
 assert abs(r["R3_mu2"]-0.340)<1e-8
