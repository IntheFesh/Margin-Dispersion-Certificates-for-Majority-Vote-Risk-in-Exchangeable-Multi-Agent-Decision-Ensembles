from src.theory.certificate import empirical_certificate_from_summaries

def test_certificate_output_fields():
 out=empirical_certificate_from_summaries(0.8,0.65,100,5,0.05)
 assert "R_cert" in out and "eta_star" in out
