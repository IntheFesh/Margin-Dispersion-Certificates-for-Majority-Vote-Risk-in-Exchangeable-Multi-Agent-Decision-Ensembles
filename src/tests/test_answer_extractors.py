from src.inference.answer_extractors import parse_multiple_choice_answer, parse_gsm8k_numeric_answer

def test_mc_parse():
    ans, inv = parse_multiple_choice_answer('The answer is (B).', ['A','B','C'])
    assert ans == 'B' and not inv

def test_gsm_parse():
    ans, inv = parse_gsm8k_numeric_answer('work... #### 1,234.50')
    assert ans == '1234.50' and not inv
